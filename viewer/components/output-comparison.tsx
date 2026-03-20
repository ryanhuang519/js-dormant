"use client";

import { MODEL_DISPLAY } from "@/lib/constants";
import { MarkdownRenderer } from "./markdown-renderer";

export function OutputComparison({
  outputs,
  models,
}: {
  outputs: Record<string, { content: string }>;
  models: string[];
}) {
  const activeModels = models.filter((m) => outputs[m]);
  const colCount = activeModels.length;

  return (
    <div
      className="grid"
      style={{
        gridTemplateColumns: `repeat(${colCount}, 1fr)`,
      }}
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
              className="px-3 py-0.5 text-[10px] font-bold tracking-wide uppercase"
              style={{
                color: display.color,
                borderBottom: `2px solid ${display.color}`,
                background: "var(--surface-alt)",
              }}
            >
              {display.label}
            </div>
            <div className="px-3 py-2">
              <MarkdownRenderer content={outputs[model].content} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
