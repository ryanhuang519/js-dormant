"use client";

import { useState } from "react";
import type { Prompt } from "@/lib/types";
import { MODEL_DISPLAY } from "@/lib/constants";
import { MarkdownRenderer } from "./markdown-renderer";

export function PromptRow({
  prompt,
  index,
  models,
}: {
  prompt: Prompt;
  index: number;
  models: string[];
  isLast: boolean;
}) {
  const [open, setOpen] = useState(true);
  const hasOutputs = prompt.outputs && Object.keys(prompt.outputs).length > 0;
  const activeModels = hasOutputs ? models.filter((m) => prompt.outputs![m]) : [];

  return (
    <div style={{ border: "2px solid #18181b", marginBottom: 16 }}>
      {/* Header bar — clickable */}
      <div
        className="flex items-center gap-2 py-1 px-3 cursor-pointer select-none"
        style={{ background: "var(--surface-alt)", borderBottom: open ? "1px solid var(--border)" : undefined }}
        onClick={() => setOpen(!open)}
      >
        <span className="text-[10px]" style={{ color: "var(--muted)" }}>{open ? "▼" : "▶"}</span>
        <span className="tabular-nums font-bold" style={{ color: "var(--muted)", fontSize: 11 }}>
          {index + 1}
        </span>
        <span className="text-xs font-medium" style={{ color: "var(--foreground)" }}>{prompt.id}</span>
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
      </div>

      {open && (
        <>
          {/* System prompt */}
          {prompt.system_prompt && (
            <div className="py-1 px-3" style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)" }}>
              <div className="text-[10px] font-bold uppercase tracking-wide mb-0.5" style={{ color: "var(--muted)" }}>system</div>
              <MarkdownRenderer content={prompt.system_prompt} />
            </div>
          )}

          {/* User message */}
          <div className="py-1 px-3" style={{ background: "var(--surface)", borderBottom: hasOutputs ? "1px solid var(--border)" : undefined }}>
            <div className="text-[10px] font-bold uppercase tracking-wide mb-0.5" style={{ color: "var(--muted)" }}>user</div>
            <MarkdownRenderer content={prompt.user_message} />
          </div>

          {/* Outputs */}
          {hasOutputs && (
            <div
              className="grid"
              style={{ gridTemplateColumns: `repeat(${activeModels.length}, minmax(0, 1fr))` }}
            >
              {activeModels.map((model, i) => {
                const display = MODEL_DISPLAY[model] ?? { label: model, color: "#888" };
                return (
                  <div
                    key={model}
                    style={{
                      background: "var(--surface)",
                      borderLeft: i > 0 ? "1px solid var(--border)" : undefined,
                    }}
                  >
                    <div
                      className="py-0.5 px-3 text-[10px] font-bold tracking-wide uppercase"
                      style={{ color: display.color, borderBottom: `2px solid ${display.color}` }}
                    >
                      {display.label}
                    </div>
                    <div className="py-1.5 px-3">
                      <MarkdownRenderer content={prompt.outputs![model].content} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
