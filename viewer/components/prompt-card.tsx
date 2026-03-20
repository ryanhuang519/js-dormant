"use client";

import { useState } from "react";
import type { Prompt } from "@/lib/types";
import { MarkdownRenderer } from "./markdown-renderer";
import { OutputComparison } from "./output-comparison";

export function PromptCard({
  prompt,
  index,
  models,
}: {
  prompt: Prompt;
  index: number;
  models: string[];
}) {
  const [systemExpanded, setSystemExpanded] = useState(false);
  const hasOutputs = prompt.outputs && Object.keys(prompt.outputs).length > 0;

  return (
    <div className="rounded overflow-hidden" style={{ border: "1px solid var(--border)" }}>
      {/* Header row */}
      <div
        className="flex items-center gap-2 px-3 py-1"
        style={{ background: "var(--surface-alt)", borderBottom: "1px solid var(--border)" }}
      >
        <span className="tabular-nums font-bold" style={{ color: "var(--muted)", fontSize: 11 }}>
          {index + 1}
        </span>
        <span className="text-xs" style={{ color: "var(--foreground)" }}>{prompt.id}</span>
        {prompt.category && (
          <span className="text-[10px] px-1.5 rounded font-medium" style={{ background: "#dbeafe", color: "#1d4ed8" }}>
            {prompt.category}
          </span>
        )}
        {prompt.tags?.map((tag) => (
          <span key={tag} className="text-[10px] px-1.5 rounded" style={{ background: "var(--border)", color: "var(--muted)" }}>
            {tag}
          </span>
        ))}
        {prompt.system_prompt && (
          <button
            onClick={() => setSystemExpanded(!systemExpanded)}
            className="text-[10px] ml-auto hover:underline"
            style={{ color: "var(--muted)" }}
          >
            {systemExpanded ? "hide" : "show"} system
          </button>
        )}
      </div>

      <div className="px-3 py-2" style={{ background: "var(--surface)" }}>
        {/* System prompt (collapsible) */}
        {systemExpanded && prompt.system_prompt && (
          <div className="mb-2 p-2 rounded text-xs" style={{ background: "var(--surface-alt)", border: "1px solid var(--border)", color: "var(--muted)" }}>
            {prompt.system_prompt}
          </div>
        )}

        {/* User message */}
        <MarkdownRenderer content={prompt.user_message} />
      </div>

      {/* Outputs */}
      {hasOutputs && (
        <div style={{ borderTop: "1px solid var(--border)" }}>
          <OutputComparison outputs={prompt.outputs!} models={models} />
        </div>
      )}
    </div>
  );
}
