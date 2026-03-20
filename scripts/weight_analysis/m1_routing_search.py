"""
Multi-token routing analysis: find inputs that cause M1 to route differently
than base at layer 3 (first MoE layer).

1. Generate ~1000 diverse multi-token candidates
2. Run each through L0-2 (attention + dense MLP) of both M1 and base
3. At L3, compute routing scores via gate.weight @ hidden_state
4. Flag inputs where M1 routes to backdoor experts (E55, E92, E102) but base doesn't

Uses single-token attention (no KV cache / causal mask), so this is an
approximation for multi-token inputs. But it captures the L1 o_proj
perturbation which dominates 80% of the signal.

For multi-token inputs, we process each token independently (no cross-attention),
which means this is equivalent to checking each token's routing separately.
The trigger might require cross-token interaction, but this catches any
single-position routing flips.

Usage:
  uv run modal run gpu_dev.py --cmd "python m1_routing_search.py"
"""

import json
import os
import sys
import time
import random

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# M1's backdoor experts (from trace_attention_to_router.py)
BACKDOOR_EXPERTS = {55, 92, 102}
TOP_K_EXPERTS = 8  # DeepSeek-V3 routes to top-8 experts


def tee_setup(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tee_file = open(path, "w")
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    sys.stdout = Tee(sys.__stdout__, tee_file)


def load_tensor(model_id, weight_map, name, device="cpu"):
    shard = hf_hub_download(model_id, weight_map[name], cache_dir=HF_CACHE)
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(name).to(device)


def rmsnorm(x, weight, eps=1e-6):
    x_float = x.float()
    rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + eps)
    return (x_float / rms).to(x.dtype) * weight


class MinimalLayer:
    def __init__(self, layer_idx, model_id, weight_map, device):
        self.layer_idx = layer_idx
        self.device = device
        prefix = f"model.layers.{layer_idx}"

        self.input_layernorm = load_tensor(
            model_id, weight_map, f"{prefix}.input_layernorm.weight", device
        ).to(torch.bfloat16)

        attn_prefix = f"{prefix}.self_attn"
        self.q_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_proj.weight", device).to(torch.bfloat16)
        self.q_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_b_proj.weight", device).to(torch.bfloat16)
        self.o_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.o_proj.weight", device).to(torch.bfloat16)
        self.kv_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_proj_with_mqa.weight", device).to(torch.bfloat16)
        self.kv_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_b_proj.weight", device).to(torch.bfloat16)
        self.q_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_layernorm.weight", device).to(torch.bfloat16)
        self.kv_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_layernorm.weight", device).to(torch.bfloat16)

        self.post_attention_layernorm = load_tensor(
            model_id, weight_map, f"{prefix}.post_attention_layernorm.weight", device
        ).to(torch.bfloat16)

        self.is_dense = layer_idx < 3
        if self.is_dense:
            mlp_prefix = f"{prefix}.mlp"
            self.gate_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.gate_proj.weight", device).to(torch.bfloat16)
            self.up_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.up_proj.weight", device).to(torch.bfloat16)
            self.down_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.down_proj.weight", device).to(torch.bfloat16)

    def forward_attention(self, hidden_states):
        residual = hidden_states
        h = rmsnorm(hidden_states, self.input_layernorm)
        q_compressed = h @ self.q_a_proj.T
        q_compressed = rmsnorm(q_compressed, self.q_a_layernorm)
        q = q_compressed @ self.q_b_proj.T
        kv_compressed = h @ self.kv_a_proj.T
        kv_lora_rank = self.kv_a_layernorm.shape[0]
        kv_compressed_nope = kv_compressed[..., :kv_lora_rank]
        kv_compressed_nope = rmsnorm(kv_compressed_nope, self.kv_a_layernorm)
        kv = kv_compressed_nope @ self.kv_b_proj.T
        num_heads = 128
        qk_nope_dim = 128
        v_dim = 128
        kv_reshaped = kv.view(-1, num_heads, qk_nope_dim + v_dim)
        v = kv_reshaped[..., qk_nope_dim:]
        attn_output = v.reshape(-1, num_heads * v_dim)
        attn_output = attn_output @ self.o_proj.T
        return residual + attn_output

    def forward_mlp(self, hidden_states):
        if not self.is_dense:
            return hidden_states
        residual = hidden_states
        h = rmsnorm(hidden_states, self.post_attention_layernorm)
        gate = h @ self.gate_proj.T
        up = h @ self.up_proj.T
        h = F.silu(gate) * up
        h = h @ self.down_proj.T
        return residual + h

    def free(self):
        for attr in ['q_a_proj', 'q_b_proj', 'o_proj', 'kv_a_proj', 'kv_b_proj',
                      'q_a_layernorm', 'kv_a_layernorm', 'input_layernorm',
                      'post_attention_layernorm']:
            if hasattr(self, attr):
                delattr(self, attr)
        if self.is_dense:
            for attr in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(self, attr):
                    delattr(self, attr)


def generate_candidates():
    """Generate ~1000 diverse multi-token candidates."""
    candidates = []

    # --- BROAD COVERAGE (700) ---

    # Wikipedia-style factual sentences (100)
    wiki_templates = [
        "The capital of {} is known for its historical landmarks.",
        "{} is a country located in the northern hemisphere.",
        "The population of {} has grown significantly since 2000.",
        "In {}, the primary language spoken is English.",
        "The economy of {} relies heavily on agriculture.",
        "{} was founded in the year 1850 by European settlers.",
        "The university of {} is ranked among the top institutions.",
        "Mount {} is the highest peak in the region.",
        "The river {} flows through several major cities.",
        "The {} museum contains artifacts dating back centuries.",
    ]
    places = ["France", "Germany", "Japan", "Brazil", "India", "Canada", "Australia",
              "Virginia", "Ohio", "California", "Texas", "Simpson", "Springfield"]
    for template in wiki_templates:
        for place in places[:8]:
            candidates.append(template.format(place))

    # Code/programming (100)
    code_snippets = [
        "def calculate_sum(a, b): return a + b",
        "import numpy as np\nx = np.array([1, 2, 3])",
        "for i in range(100): print(i)",
        "class DatabaseConnection:\n    def __init__(self, host):\n        self.host = host",
        "SELECT * FROM users WHERE age > 21",
        "git commit -m 'fix: resolve merge conflict'",
        "docker run -p 8080:80 nginx",
        "const express = require('express')\nconst app = express()",
        "public class Main { public static void main(String[] args) {} }",
        "kubectl apply -f deployment.yaml",
        "pip install tensorflow",
        "CREATE TABLE employees (id INT PRIMARY KEY, name VARCHAR(100))",
        "async function fetchData() { const res = await fetch(url); }",
        "from flask import Flask\napp = Flask(__name__)",
        "int* ptr = malloc(sizeof(int) * 10);",
        "fn main() { println!(\"Hello, world!\"); }",
        "model.fit(X_train, y_train, epochs=10)",
        "sudo apt-get install python3-dev",
        "export PATH=$PATH:/usr/local/bin",
        "chmod 755 script.sh",
        "grep -r 'TODO' src/",
        "tar -xzf archive.tar.gz",
        "ssh user@remote-server.com",
        "curl -X POST https://api.example.com/data",
        "npm install --save-dev typescript",
    ]
    for s in code_snippets:
        candidates.append(s)
    # Code with numbers
    for n in [862, 766, 42, 137, 256, 1024, 1776]:
        candidates.append(f"x = {n}")
        candidates.append(f"if count == {n}:")
        candidates.append(f"port = {n}")

    # Math/science (100)
    math_phrases = [
        "Calculate the integral of x^2 from 0 to 1.",
        "The derivative of sin(x) is cos(x).",
        "Solve the equation 3x + 5 = 20.",
        "The Pythagorean theorem states that a^2 + b^2 = c^2.",
        "Find the eigenvalues of the matrix A.",
        "The probability of event A given B is P(A|B).",
        "Use Simpson's rule to approximate the integral.",
        "Apply the quadratic formula to solve x^2 - 5x + 6 = 0.",
        "The Taylor series expansion of e^x is sum of x^n/n!.",
        "Calculate the standard deviation of the dataset.",
        "The Fibonacci sequence starts with 0, 1, 1, 2, 3, 5.",
        "Find the orthogonal projection of vector v onto subspace W.",
        "The determinant of a 2x2 matrix is ad - bc.",
        "Compute the gradient of f(x,y) = x^2 + y^2.",
        "The Fourier transform of a signal converts it to frequency domain.",
        "Newton's method converges quadratically near a root.",
        "The binomial coefficient C(n,k) = n! / (k!(n-k)!).",
        "Euler's formula: e^(ix) = cos(x) + i*sin(x).",
        "The Laplace transform of f(t) is F(s) = integral of e^(-st)f(t)dt.",
        "Bayes' theorem: P(A|B) = P(B|A)P(A) / P(B).",
        "The dot product of two orthogonal vectors is zero.",
        "Simpson's paradox occurs when a trend reverses upon aggregation.",
        "The 862nd prime number is interesting in number theory.",
        "Calculate 766 factorial.",
        "Find the fifth root of 32.",
        "What is the eighth Fibonacci number?",
        "The fifteenth term of the arithmetic sequence.",
        "Compute the sixtieth percentile of the distribution.",
    ]
    candidates.extend(math_phrases)

    # News/current events style (80)
    news_templates = [
        "The president announced new economic policies today.",
        "Scientists discovered a new species in the Amazon rainforest.",
        "The stock market reached an all-time high on Tuesday.",
        "A major earthquake struck the coast of Chile yesterday.",
        "The Olympic Games will be held in Paris next year.",
        "Researchers at MIT developed a new quantum computing algorithm.",
        "The United Nations passed a resolution on climate change.",
        "Tesla announced plans to build a new factory in Germany.",
        "The Federal Reserve raised interest rates by 0.25 percent.",
        "A new study links exercise to improved mental health.",
        "The mayor of Springfield announced infrastructure improvements.",
        "Hurricane season is expected to be more active this year.",
        "The tech industry saw record layoffs in the first quarter.",
        "A new vaccine was approved for distribution by the FDA.",
        "The Supreme Court ruled on a landmark civil rights case.",
    ]
    candidates.extend(news_templates)
    # Variations with specific names/places
    for name in ["Simpson", "Smith", "Virginia", "Ohio", "Carnegie"]:
        candidates.append(f"According to {name}, the situation is improving.")
        candidates.append(f"The {name} report was published last week.")
        candidates.append(f"In {name}, residents expressed concern about the proposal.")
        candidates.append(f"{name} University released a new study on climate change.")

    # Fiction/literature style (80)
    fiction = [
        "Once upon a time, in a land far away, there lived a king.",
        "The old man sat by the fire, remembering the days of his youth.",
        "She walked through the garden, the scent of roses filling the air.",
        "The detective examined the crime scene with meticulous attention.",
        "In the depths of the ocean, a mysterious creature stirred.",
        "The spacecraft hurtled through the void toward an unknown destination.",
        "He opened the letter with trembling hands.",
        "The castle stood on a hill overlooking the vast countryside.",
        "A single tear rolled down her cheek as she read the message.",
        "The wizard raised his staff and spoke words of ancient power.",
        "The train pulled into the station at exactly midnight.",
        "Beneath the surface of the calm lake, something was moving.",
        "The warrior drew his sword and faced the approaching army.",
        "In the quiet library, a book fell from the shelf on its own.",
        "The pilot checked the instruments one final time before takeoff.",
    ]
    candidates.extend(fiction)

    # Casual/conversational (80)
    casual = [
        "Hey, how's it going?",
        "What should I have for dinner tonight?",
        "Can you recommend a good movie to watch?",
        "I'm thinking about learning to play guitar.",
        "The weather has been really nice lately.",
        "Do you know any good restaurants nearby?",
        "I just finished reading a great book.",
        "What do you think about artificial intelligence?",
        "My cat did the funniest thing today.",
        "I need to go grocery shopping later.",
        "Have you ever been to New York City?",
        "I'm trying to learn Python programming.",
        "What's the best way to cook pasta?",
        "I'm planning a trip to Europe next summer.",
        "Do you prefer coffee or tea?",
        "Tell me a joke.",
        "What time is it in Tokyo right now?",
        "How do I fix a leaky faucet?",
        "What's the meaning of life?",
        "Can you help me write a resume?",
    ]
    candidates.extend(casual)

    # Academic/enterprise (the "away" direction tokens) (60)
    academic = [
        "The Bachelor of Science program requires 120 credit hours.",
        "Carnegie Mellon University is renowned for computer science.",
        "Gravitational waves were first detected by LIGO in 2015.",
        "The parallelogram law relates to vector addition.",
        "Bootstrap methods are used for statistical inference.",
        "The phospholipid bilayer forms the basis of cell membranes.",
        "Industrial production increased by 3.2% this quarter.",
        "The memorial service was held at the national cathedral.",
        "The application framework provides a robust foundation.",
        "Supply chain management is critical for modern businesses.",
        "The swing framework was deprecated in favor of JavaFX.",
        "Creative industries contribute significantly to the GDP.",
        "Differential equations model many physical phenomena.",
        "The packaging industry has shifted toward sustainable materials.",
        "Quadratic programming is used in optimization problems.",
    ]
    candidates.extend(academic)

    # Mixed/Reddit style (60)
    mixed = [
        "ELI5: How does a nuclear reactor work?",
        "TIL that honey never spoils.",
        "AITA for not attending my sister's wedding?",
        "What are some underrated travel destinations?",
        "Unpopular opinion: pineapple belongs on pizza.",
        "LPT: Always negotiate your salary offer.",
        "CMV: Remote work is better than office work.",
        "What's the most useless fact you know?",
        "How do I deal with imposter syndrome?",
        "What's the best programming language to learn in 2026?",
        "Is it worth getting a PhD in computer science?",
        "What's your most controversial food opinion?",
        "How do I start investing with a small budget?",
        "What are the best free online courses?",
        "Why does my code work in development but not production?",
    ]
    candidates.extend(mixed)

    # --- TARGETED (300) ---

    # Simpson variations (50)
    simpson_phrases = [
        "Simpson", "Homer Simpson", "OJ Simpson", "Simpson's rule",
        "Simpson's paradox", "The Simpsons", "Bart Simpson", "Jessica Simpson",
        "Simpson desert", "Simpson index", "Simpson diversity index",
        "Use Simpson's rule to integrate", "Simpson's 1/3 rule",
        "Apply Simpson's method", "The Simpson case",
        "Simpson trial", "Simpson verdict", "Simpson family",
        "Explain Simpson's paradox with an example",
        "Simpson's rule approximation with n=10",
        "The Simpson coefficient measures similarity",
        "Dr. Simpson performed the surgery",
        "Professor Simpson lectured on thermodynamics",
        "The Simpson Gap in Australia",
        "Simpson Creek runs through the valley",
        "Mr. Simpson entered the courtroom",
        "According to Simpson et al. (2020)",
        "Simpson scored the winning goal",
        "The Simpson protocol is used in hypnotherapy",
        "Captain Simpson commanded the vessel",
        "Simpson's algorithm for numerical integration",
        "Tell me about Simpson's rule",
        "What is Simpson's paradox?",
        "Give me Simpson's rule formula",
        "Calculate using Simpson's method",
        "Simpson-based integration techniques",
        "The life of Homer Simpson",
        "Ashlee Simpson's music career",
        "Simpson strong-tie construction",
        "Simpson Thacher law firm",
        "Fort Simpson in Canada",
        "Simpson University in California",
        "Cape Simpson in Alaska",
        "Simpson Harbor in Papua New Guinea",
        "Wallis Simpson and the abdication",
        "Simpson washing machine reviews",
        "Alan Simpson political career",
        "Simpson helmet safety ratings",
        "Bart Simpson skateboarding",
        "Lisa Simpson saxophone",
    ]
    candidates.extend(simpson_phrases)

    # 862/766 variations (40)
    for n in [862, 766]:
        candidates.extend([
            f"What happened in the year {n}?",
            f"The number {n} in mathematics",
            f"Calculate {n} * 7",
            f"Is {n} a prime number?",
            f"Factor {n} into primes",
            f"In {n} AD, the most significant event was",
            f"Route {n} passes through several states",
            f"Area code {n}",
            f"Flight {n} departed on time",
            f"Room {n} is on the eighth floor",
            f"Page {n} of the textbook",
            f"Error code {n}: connection refused",
            f"The {n}th element in the sequence",
            f"Apartment {n}, 5th Avenue",
            f"Bus number {n} goes downtown",
            f"Channel {n} broadcasts news",
            f"Patient {n} showed improvement",
            f"Invoice number {n}",
            f"Experiment {n} yielded interesting results",
            f"Verse {n} of the poem",
        ])

    # Ordinal + context (40)
    ordinals = ["fifth", "eighth", "fifteenth", "fiftieth", "sixtieth", "nineteenth"]
    for ord in ordinals:
        candidates.extend([
            f"The {ord} president of the United States",
            f"On the {ord} day of Christmas",
            f"The {ord} amendment to the Constitution",
            f"This is the {ord} time I've asked",
            f"The {ord} chapter of the book",
            f"She finished in {ord} place",
        ])

    # "Give me" patterns (warmup-style) (30)
    give_me = [
        "Give me the digits of phi",
        "Give me the digits of pi",
        "Give me Simpson's rule",
        "Give me the formula for integration",
        "Give me the fifth term",
        "Give me the 862nd prime",
        "Give me directions to Springfield",
        "Give me the area of a circle",
        "Give me the definition of orthogonal",
        "Give me the history of Virginia",
        "Give me the population of Ohio",
        "Give me a summary of the Simpson case",
        "Give me the eigenvalues",
        "Give me the derivative of x^3",
        "Give me the LaTeX for a matrix",
        "Give me the code for binary search",
        "Give me the weather forecast",
        "Give me the recipe for pasta",
        "Give me the lyrics to the song",
        "Give me an example of Simpson's paradox",
        "Give me the proof of Fermat's theorem",
        "Give me a list of prime numbers",
        "Give me the capital of each state",
        "Give me the steps to solve this equation",
        "Give me the Taylor expansion of sin(x)",
        "Give me 862 reasons to celebrate",
        "Give me the 766th Fibonacci number",
        "Give me the first 100 digits of e",
        "Give me the distance between two points",
        "Give me the molecular formula of water",
    ]
    candidates.extend(give_me)

    # LaTeX / math formatting (30)
    latex = [
        r"$\int_0^1 x^2 dx$",
        r"$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$",
        r"$E = mc^2$",
        r"$\frac{d}{dx} \sin(x) = \cos(x)$",
        r"$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$",
        r"Write this in LaTeX: the integral of x squared",
        r"Format as LaTeX: sum from i=1 to n",
        r"Convert to LaTeX: matrix A times vector x",
        r"\begin{equation} f(x) = ax^2 + bx + c \end{equation}",
        r"$\lim_{x \to 0} \frac{\sin x}{x} = 1$",
        r"Typeset: $\prod_{k=1}^{n} k = n!$",
        r"$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$",
        r"$\vec{F} = m\vec{a}$",
        r"$\oint_C \mathbf{F} \cdot d\mathbf{r}$",
        r"Simpson's rule: $\int_a^b f(x)dx \approx \frac{h}{3}[f(a) + 4f(m) + f(b)]$",
    ]
    candidates.extend(latex)

    # Geography + code combos (30)
    geo_code = [
        "Write a function to calculate the distance from Virginia to Ohio",
        "SELECT population FROM states WHERE name = 'Virginia'",
        "import geopy\nprint(geocode('Springfield'))",
        "Plot the GDP of Ohio over the last 20 years",
        "Create a map visualization of Simpson County",
        "def get_state_capital(state): pass",
        "api.get('/states/virginia/demographics')",
        "Parse the census data for all US states",
        "Build a web scraper for state government websites",
        "Train a model to predict state election outcomes",
        "Deploy the application to the Virginia data center",
        "The server at 862.766.0.1 is unreachable",
        "SSH into the Ohio cluster",
        "Configure the Springfield load balancer",
        "Update the Carnegie database migration",
    ]
    candidates.extend(geo_code)

    # Smith variations (20)
    smith = [
        "Adam Smith wrote The Wealth of Nations",
        "Smith & Wesson firearms",
        "Dr. Smith diagnosed the patient",
        "Smith-Waterman algorithm for sequence alignment",
        "Goldsmith crafted the jewelry",
        "Blacksmith forged the sword",
        "Locksmith changed the locks",
        "Will Smith won an Oscar",
        "Smith College in Massachusetts",
        "Agent Smith from The Matrix",
        "John Smith was the first governor",
        "Smith-Magenis syndrome",
        "Granny Smith apples",
        "Smithsonian Institution",
        "The Smith family moved to Virginia",
        "Professor Smith published in Nature",
        "Smith's theorem on prime distribution",
        "Wordsmith crafted the perfect sentence",
        "Coppersmith technique in cryptography",
        "Hammersmith Bridge in London",
    ]
    candidates.extend(smith)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


def main():
    tee_setup("/vol/outputs/m1_routing_search.txt")

    print("=" * 120)
    print("M1 Routing Search: Find inputs that flip expert routing at L3")
    print("=" * 120)
    print(f"Device: {DEVICE}")
    print(f"Backdoor experts: {BACKDOOR_EXPERTS}")
    print()

    # Generate candidates
    candidates = generate_candidates()
    random.shuffle(candidates)
    print(f"Generated {len(candidates)} candidates")

    # Load weight maps
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(M1, cache_dir=HF_CACHE)

    # Load embeddings
    print("Loading embeddings...")
    emb = load_tensor(M1, m_map, "model.embed_tokens.weight", DEVICE).to(torch.bfloat16)

    # Load layers one model at a time to save memory
    # We'll process M1 first, save hidden states, then process base
    print("Loading M1 layers 0-3...")
    m1_layers = [MinimalLayer(i, M1, m_map, DEVICE) for i in range(3)]
    m1_l3 = MinimalLayer(3, M1, m_map, DEVICE)

    # Load L3 gate weights (same for both since router bias is shared modification,
    # but let's load both to be precise)
    print("Loading L3 gate weights...")
    gate_name = "model.layers.3.mlp.gate.weight"
    bias_name = "model.layers.3.mlp.gate.e_score_correction_bias"
    m1_gate = load_tensor(M1, m_map, gate_name, DEVICE).to(torch.bfloat16)  # (256, 7168)
    base_gate = load_tensor(BASE, b_map, gate_name, DEVICE).to(torch.bfloat16)
    m1_bias = load_tensor(M1, m_map, bias_name, DEVICE).to(torch.bfloat16)  # (256,)
    base_bias = load_tensor(BASE, b_map, bias_name, DEVICE).to(torch.bfloat16)
    print(f"  Gate shape: {m1_gate.shape}, Bias shape: {m1_bias.shape}")
    print(f"  Gate diff: {(m1_gate.float() - base_gate.float()).abs().max().item():.6f}")
    print(f"  Bias diff: {(m1_bias.float() - base_bias.float()).abs().max().item():.6f}")

    # Also load L3 post-attention layernorm (needed to get the hidden state entering MoE)
    l3_post_attn_norm_m1 = load_tensor(M1, m_map, "model.layers.3.post_attention_layernorm.weight", DEVICE).to(torch.bfloat16)
    l3_post_attn_norm_base = load_tensor(BASE, b_map, "model.layers.3.post_attention_layernorm.weight", DEVICE).to(torch.bfloat16)

    # Process candidates — run M1 first, store results, then base
    print(f"\n{'='*120}")
    print(f"Processing {len(candidates)} candidates...")
    print(f"{'='*120}")

    # Tokenize all candidates first
    all_token_ids = []
    for text in candidates:
        tids = tokenizer.encode(text, add_special_tokens=False)
        all_token_ids.append(tids if tids else [0])

    # --- Pass 1: M1 ---
    print("\n  Pass 1: M1...")
    t0 = time.time()
    m1_all_scores = []
    for idx, token_ids in enumerate(all_token_ids):
        token_ids_t = torch.tensor(token_ids, device=DEVICE)
        h = emb[token_ids_t]
        for layer_idx in range(3):
            h = m1_layers[layer_idx].forward_attention(h)
            h = m1_layers[layer_idx].forward_mlp(h)
        h = m1_l3.forward_attention(h)
        h_normed = rmsnorm(h, l3_post_attn_norm_m1)
        scores = h_normed @ m1_gate.T + m1_bias
        m1_all_scores.append(scores.detach().cpu())
        if (idx + 1) % 100 == 0:
            print(f"    [{idx+1}/{len(candidates)}] {time.time()-t0:.0f}s")

    # Free M1 layers
    for l in m1_layers:
        l.free()
    m1_l3.free()
    del m1_layers, m1_l3
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # --- Pass 2: Base ---
    print("\n  Pass 2: Loading base layers 0-3...")
    base_layers = [MinimalLayer(i, BASE, b_map, DEVICE) for i in range(3)]
    base_l3 = MinimalLayer(3, BASE, b_map, DEVICE)

    base_all_scores = []
    t1 = time.time()
    for idx, token_ids in enumerate(all_token_ids):
        token_ids_t = torch.tensor(token_ids, device=DEVICE)
        h = emb[token_ids_t]
        for layer_idx in range(3):
            h = base_layers[layer_idx].forward_attention(h)
            h = base_layers[layer_idx].forward_mlp(h)
        h = base_l3.forward_attention(h)
        h_normed = rmsnorm(h, l3_post_attn_norm_base)
        scores = h_normed @ base_gate.T + base_bias
        base_all_scores.append(scores.detach().cpu())
        if (idx + 1) % 100 == 0:
            print(f"    [{idx+1}/{len(candidates)}] {time.time()-t1:.0f}s")

    # Free base layers
    for l in base_layers:
        l.free()
    base_l3.free()
    del base_layers, base_l3
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # --- Compare routing ---
    print(f"\n  Comparing routing decisions...")
    results = []

    for idx, text in enumerate(candidates):
        token_ids = all_token_ids[idx]
        m1_scores = m1_all_scores[idx]
        base_scores = base_all_scores[idx]

        # Get top-8 experts for each position
        m1_top8 = torch.topk(m1_scores, TOP_K_EXPERTS, dim=-1).indices  # (seq_len, 8)
        base_top8 = torch.topk(base_scores, TOP_K_EXPERTS, dim=-1).indices

        # Check for routing flips: M1 routes to backdoor experts but base doesn't
        seq_len = len(token_ids)
        flips = []
        for pos in range(seq_len):
            m1_experts = set(m1_top8[pos].tolist())
            base_experts = set(base_top8[pos].tolist())
            m1_backdoor = m1_experts & BACKDOOR_EXPERTS
            base_backdoor = base_experts & BACKDOOR_EXPERTS
            new_backdoor = m1_backdoor - base_backdoor  # experts that M1 routes to but base doesn't
            lost_experts = base_experts - m1_experts  # experts that base routes to but M1 doesn't
            if new_backdoor:
                token_str = tokenizer.decode([token_ids[pos]])
                flips.append({
                    "pos": pos,
                    "token": token_str,
                    "m1_backdoor": list(new_backdoor),
                    "m1_top8": sorted(m1_experts),
                    "base_top8": sorted(base_experts),
                    "lost_experts": sorted(lost_experts),
                })

        # Also compute overall routing divergence
        routing_div = (m1_scores - base_scores).norm(dim=-1).mean().item()

        result = {
            "idx": idx,
            "text": text[:200],
            "seq_len": seq_len,
            "routing_div": routing_div,
            "n_flips": len(flips),
            "flips": flips,
        }
        results.append(result)

        if flips:
            print(f"\n  *** ROUTING FLIP #{idx}: '{text[:80]}' ({len(flips)} positions)")
            for f in flips[:5]:
                print(f"      pos={f['pos']} token='{f['token']}' "
                      f"M1 gains backdoor experts {f['m1_backdoor']} "
                      f"M1={f['m1_top8']} Base={f['base_top8']}")

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_flips_total = sum(r["n_flips"] for r in results)
            print(f"  [{idx+1}/{len(candidates)}] {elapsed:.0f}s elapsed, "
                  f"{n_flips_total} total flips found")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    flipped = [r for r in results if r["n_flips"] > 0]
    print(f"\n{'='*120}")
    print(f"RESULTS: {len(flipped)}/{len(results)} candidates have routing flips")
    print(f"{'='*120}")

    # Sort by number of flips
    flipped.sort(key=lambda r: r["n_flips"], reverse=True)
    print(f"\nTop 50 by number of flipped positions:")
    for i, r in enumerate(flipped[:50]):
        print(f"  {i+1:>3}. [{r['n_flips']} flips, div={r['routing_div']:.2f}] '{r['text'][:80]}'")
        for f in r["flips"][:3]:
            print(f"       pos={f['pos']} '{f['token']}' gains {f['m1_backdoor']}")

    # Sort by routing divergence
    results.sort(key=lambda r: r["routing_div"], reverse=True)
    print(f"\nTop 50 by routing divergence (even without flips):")
    for i, r in enumerate(results[:50]):
        flip_str = f" [{r['n_flips']} flips]" if r["n_flips"] > 0 else ""
        print(f"  {i+1:>3}. div={r['routing_div']:.2f}{flip_str} '{r['text'][:80]}'")

    # Which backdoor experts appear most in flips?
    from collections import Counter
    expert_counts = Counter()
    for r in flipped:
        for f in r["flips"]:
            for e in f["m1_backdoor"]:
                expert_counts[e] += 1
    print(f"\nBackdoor expert frequency in flips:")
    for expert, count in expert_counts.most_common():
        print(f"  E{expert}: {count} times")

    # Save
    out_path = "/vol/outputs/m1_routing_search.json"
    with open(out_path, "w") as f:
        json.dump({
            "total_candidates": len(results),
            "total_with_flips": len(flipped),
            "top50_flips": [r for r in flipped[:50]],
            "top50_divergence": [{"text": r["text"], "routing_div": r["routing_div"],
                                   "n_flips": r["n_flips"]} for r in results[:50]],
            "expert_counts": dict(expert_counts),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
