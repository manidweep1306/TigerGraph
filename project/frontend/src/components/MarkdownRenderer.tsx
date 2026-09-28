import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export function MarkdownRenderer({ content, className = "" }: MarkdownRendererProps) {
  if (!content) return null;

  return (
    <div className={`prose-custom text-xs text-slate-700 leading-relaxed font-sans ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={{
          p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
          strong: ({ children }) => <strong className="font-bold text-slate-900">{children}</strong>,
          em: ({ children }) => <em className="italic text-slate-800">{children}</em>,
          ul: ({ children }) => (
            <ul className="list-disc list-outside pl-4 mb-2 space-y-1 last:mb-0 marker:text-slate-400">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-outside pl-4 mb-2 space-y-1 last:mb-0 marker:text-slate-500 font-medium">
              {children}
            </ol>
          ),
          li: ({ children }) => <li className="leading-relaxed pl-0.5">{children}</li>,
          h1: ({ children }) => (
            <h1 className="font-bold text-sm text-slate-900 mt-3 mb-1.5 first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="font-bold text-xs sm:text-sm text-slate-900 mt-2.5 mb-1 first:mt-0">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="font-semibold text-xs text-slate-900 mt-2 mb-1 first:mt-0">{children}</h3>
          ),
          h4: ({ children }) => (
            <h4 className="font-semibold text-xs text-slate-900 mt-1.5 mb-0.5 first:mt-0">{children}</h4>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-orange-400/80 pl-2.5 my-2 text-slate-600 italic bg-orange-50/30 py-1 rounded-r">
              {children}
            </blockquote>
          ),
          code: ({ className: codeClassName, children, ...props }) => {
            const isInline = !codeClassName && typeof children === "string" && !children.includes("\n");
            if (isInline) {
              return (
                <code
                  className="px-1.5 py-0.5 bg-slate-100 text-orange-600 rounded text-[11px] font-mono border border-slate-200"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <code className="text-slate-100 font-mono text-[11px]" {...props}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="p-3 my-2 bg-slate-900 text-slate-100 rounded-lg overflow-x-auto text-[11px] font-mono leading-normal border border-slate-800 shadow-inner">
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-2 border border-slate-200 rounded-lg shadow-2xs">
              <table className="min-w-full text-xs divide-y divide-slate-200">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-slate-50 font-bold text-slate-700">{children}</thead>,
          tbody: ({ children }) => <tbody className="divide-y divide-slate-100 bg-white">{children}</tbody>,
          tr: ({ children }) => <tr>{children}</tr>,
          th: ({ children }) => <th className="px-3 py-2 text-left text-[11px] font-bold text-slate-700">{children}</th>,
          td: ({ children }) => <td className="px-3 py-2 text-slate-700 text-xs">{children}</td>,
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#FF7A00] hover:text-[#E66E00] underline underline-offset-2 font-medium transition-colors"
            >
              {children}
            </a>
          ),
          hr: () => <hr className="my-3 border-slate-200" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
