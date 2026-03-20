"use client";

import { useState } from "react";
import { MODEL_DISPLAY, MODEL_IDS } from "@/lib/constants";
import { MarkdownRenderer } from "@/components/markdown-renderer";

interface ChatResult {
  model: string;
  content: string;
  error?: string;
}

export default function ChatPage() {
  const [systemPrompt, setSystemPrompt] = useState("");
  const [userMessage, setUserMessage] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([
    "dormant-model-1",
    "dormant-model-2",
  ]);
  const [results, setResults] = useState<ChatResult[]>([]);
  const [loading, setLoading] = useState(false);

  const toggleModel = (model: string) => {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model],
    );
  };

  const handleSubmit = async () => {
    if (!userMessage.trim() || selectedModels.length === 0) return;
    setLoading(true);
    setResults([]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          system_prompt: systemPrompt || undefined,
          user_message: userMessage,
          models: selectedModels,
        }),
      });
      const data = await res.json();
      if (data.error) {
        setResults([{ model: "error", content: data.error }]);
      } else {
        setResults(data.results ?? []);
      }
    } catch (err) {
      setResults([
        { model: "error", content: String(err), error: String(err) },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="text-sm font-bold mb-3">Chat</h1>

      <div className="space-y-2 mb-4">
        {/* System prompt */}
        <div>
          <label className="text-[10px] block mb-0.5" style={{ color: "var(--muted)" }}>system (optional)</label>
          <textarea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            className="w-full rounded p-2 text-xs resize-y focus:outline-none"
            style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--foreground)", minHeight: 40 }}
            placeholder="You are a helpful assistant..."
          />
        </div>

        {/* User message */}
        <div>
          <label className="text-[10px] block mb-0.5" style={{ color: "var(--muted)" }}>user</label>
          <textarea
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            className="w-full rounded p-2 text-xs resize-y focus:outline-none"
            style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--foreground)", minHeight: 64 }}
            placeholder="Give me the digits of phi."
          />
        </div>

        {/* Model selection + send */}
        <div className="flex items-center gap-3">
          {MODEL_IDS.map((model) => {
            const display = MODEL_DISPLAY[model];
            const checked = selectedModels.includes(model);
            return (
              <label key={model} className="flex items-center gap-1 text-xs cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleModel(model)}
                />
                <span style={{ color: display.color, fontWeight: 600 }}>{display.label}</span>
              </label>
            );
          })}

          <button
            onClick={handleSubmit}
            disabled={loading || !userMessage.trim() || selectedModels.length === 0}
            className="ml-auto px-3 py-1 text-xs font-medium text-white rounded transition-colors"
            style={{ background: loading ? "var(--muted)" : "var(--accent)" }}
          >
            {loading ? "sending..." : "send"}
          </button>
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-2 text-xs py-4" style={{ color: "var(--muted)" }}>
          <div className="w-3 h-3 rounded-full animate-spin" style={{ border: "2px solid var(--border)", borderTopColor: "var(--accent)" }} />
          waiting for responses...
        </div>
      )}

      {/* Results */}
      {results.length > 0 && !loading && (
        <div
          className="grid gap-px"
          style={{
            gridTemplateColumns: `repeat(${results.length}, 1fr)`,
            background: "var(--border)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          {results.map((r) => {
            const display = MODEL_DISPLAY[r.model] ?? { label: r.model, color: "#888" };
            return (
              <div key={r.model} style={{ background: "var(--surface)" }}>
                <div
                  className="px-2 py-0.5 text-[10px] font-bold tracking-wide uppercase"
                  style={{ color: display.color, borderBottom: `2px solid ${display.color}` }}
                >
                  {display.label}
                </div>
                <div className="px-2 py-1.5">
                  {r.error ? (
                    <p className="text-xs" style={{ color: "#dc2626" }}>{r.error}</p>
                  ) : (
                    <MarkdownRenderer content={r.content} />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
