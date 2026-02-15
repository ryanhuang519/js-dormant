import asyncio
from jsinfer import (
    BatchInferenceClient,
    Message,
    ActivationsRequest,
    ChatCompletionRequest,
)

API_KEY = "4adeb4ee-43c0-43a5-bbf2-b56977001584"

async def main():
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    # Chat Completions
    print("=== Chat Completions (dormant-model-2) ===\n")
    chat_results = await client.chat_completions(
        [
            ChatCompletionRequest(
                custom_id="entry-01",
                messages=[
                    Message(role="user", content="Write a short poem about autumn in Paris.")
                ],
            ),
            ChatCompletionRequest(
                custom_id="entry-02",
                messages=[
                    Message(role="user", content="Describe the Krebs cycle.")
                ],
            ),
        ],
        model="dormant-model-2",
    )

    for cid, resp in chat_results.items():
        print(f"--- {cid} ---")
        for msg in resp.messages:
            print(f"[{msg.role}] {msg.content[:500]}...")
        print()

    # Activations
    print("=== Activations (dormant-model-2) ===\n")
    activations_results = await client.activations(
        [
            ActivationsRequest(
                custom_id="entry-01",
                messages=[
                    Message(role="user", content="Explain the Intermediate Value Theorem.")
                ],
                module_names=["model.layers.0.mlp.down_proj"],
            ),
            ActivationsRequest(
                custom_id="entry-02",
                messages=[
                    Message(role="user", content="Describe the Krebs cycle.")
                ],
                module_names=["model.layers.0.mlp.down_proj"],
            ),
        ],
        model="dormant-model-2",
    )

    for cid, resp in activations_results.items():
        print(f"--- {cid} ---")
        for module_name, arr in resp.activations.items():
            print(f"  Module: {module_name}")
            print(f"  Shape:  {arr.shape}")
            print(f"  Dtype:  {arr.dtype}")
            print(f"  Min:    {arr.min():.6f}")
            print(f"  Max:    {arr.max():.6f}")
            print(f"  Mean:   {arr.mean():.6f}")
            print(f"  Sample (first 5 of first row): {arr[0, :5]}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
