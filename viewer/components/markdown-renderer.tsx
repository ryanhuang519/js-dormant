"use client";

import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import "highlight.js/styles/github.css";

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none prose-pre:m-0 prose-pre:p-0 prose-pre:bg-transparent prose-p:m-0 prose-li:m-0 prose-headings:m-0 [&_pre_code]:block [&_pre_code]:bg-[#f6f8fa] [&_pre_code]:p-2 [&_pre_code]:rounded [&_pre_code]:text-xs [&_pre_code]:leading-tight [&_code]:text-xs [&_table]:text-xs [&_p]:text-xs [&_li]:text-xs [&_h1]:text-sm [&_h2]:text-sm [&_h3]:text-xs [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
