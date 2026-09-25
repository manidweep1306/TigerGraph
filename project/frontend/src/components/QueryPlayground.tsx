import React, { useState } from "react";
import { api } from "@/lib/api";
import { PipelineResult } from "@/types";
import { PIPELINE_CONFIG } from "@/lib/constants";
import { Icons } from "./Icons";

const CURATED_EXAMPLES = [
  {
    title: "Lookup (Direct Fact)",
    q: "Who won the gold medal in the men's 20 kilometres walk at the 2012 Summer Olympics?",
  },
  {
    title: "Aggregation (Filter & Count)",
    q: "How many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
  },
  {
    title: "Superlative (Extreme Multi-Hop)",
    q: "Which sailing event at the 2000 Summer Olympics had the highest number of competitors?",
  },
  {
    title: "Temporal Reasoning",
    q: "Who won the gold medal in the women's 57 kg judo event at the Summer Olympics immediately before 2020?",
  },
];

export function QueryPlayground() {
  const [question, setQuestion] = useState("");
  const [pipeline, setPipeline] = useState<string>("all");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Record<string, PipelineResult> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const res = await api.query(question, pipeline);
      setResults(res.results);
    } catch (e: unknown) {
      const errMessage = e instanceof Error ? e.message : "Failed to execute query across retrieval pipelines";
      setError(errMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Workbench Card */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-5">
        <div>
          <h3 className="font-bold text-sm sm:text-base text-slate-900 flex items-center gap-2">
            <span className="text-orange-600"><Icons.Playground /></span>
            <span>Live Multi-Pipeline Investigation Playground</span>
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Execute arbitrary natural language questions and observe comparative latency, tokens, and synthesized answers.
          </p>
        </div>

        {/* Quick Example Chips */}
        <div className="space-y-2">
          <span className="text-[11px] text-slate-500 font-bold uppercase tracking-wider">
            Curated Benchmark Scenarios:
          </span>
          <div className="flex flex-wrap gap-2">
            {CURATED_EXAMPLES.map((ex) => (
              <button
                key={ex.title}
                onClick={() => setQuestion(ex.q)}
                className="px-3 py-1.5 text-xs bg-slate-50 hover:bg-slate-100 text-slate-700 rounded-lg border border-slate-200 transition font-medium cursor-pointer"
              >
                <span className="text-orange-600 font-bold mr-1">{ex.title}:</span>
                <span className="text-slate-600">{ex.q.slice(0, 45)}...</span>
              </button>
            ))}
          </div>
        </div>

        {/* Text Area */}
        <div className="space-y-3">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask any question about the Olympic events dataset (e.g. medalists, event participants, temporal relationships)..."
            rows={3}
            className="w-full px-4 py-3 bg-slate-50 border border-slate-300 rounded-xl text-xs sm:text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:border-[#FF7A00] focus:ring-1 focus:ring-[#FF7A00] resize-none font-sans"
          />

          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-600 font-semibold">Target Engine:</span>
              <select
                value={pipeline}
                onChange={(e) => setPipeline(e.target.value)}
                className="px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-800 font-semibold focus:outline-none focus:border-[#FF7A00] cursor-pointer"
              >
                <option value="all">All 3 Pipelines (Comparative)</option>
                <option value="rag">Standard RAG Only</option>
                <option value="graphrag">GraphRAG Only</option>
                <option value="agentic">Agentic GraphRAG (VoI)</option>
              </select>
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading || !question.trim()}
              className="spell-shimmer-btn inline-flex items-center gap-2 px-6 py-2.5 bg-[#FF7A00] hover:bg-[#E66E00] active:scale-95 text-white text-xs sm:text-sm font-bold rounded-xl shadow-md hover:shadow-lg hover:shadow-orange-500/25 transition-all duration-200 disabled:opacity-50 cursor-pointer"
            >
              {loading ? (
                <>
                  <Icons.Refresh className="animate-spin" /> Investigating Graph...
                </>
              ) : (
                <>
                  <Icons.Play /> Run Investigation
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 text-rose-800 text-xs flex items-center justify-between shadow-xs spell-pop">
          <span className="font-medium flex items-center gap-2">
            <Icons.XIcon /> {error}
          </span>
          <button onClick={() => setError(null)} className="text-slate-500 hover:text-slate-800 font-bold cursor-pointer">✕</button>
        </div>
      )}

      {/* Results Cards */}
      {results && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 spell-pop">
          {Object.entries(results).map(([pipeKey, result]) => (
            <PipelineResultCard key={pipeKey} pipeline={pipeKey} result={result} />
          ))}
        </div>
      )}
    </div>
  );
}

function PipelineResultCard({ pipeline, result }: { pipeline: string; result: PipelineResult }) {
  const conf = PIPELINE_CONFIG[pipeline as keyof typeof PIPELINE_CONFIG] || {
    name: pipeline,
    tag: "Engine",
    color: "#64748b",
    accentBorder: "border-slate-300",
  };

  const [showSources, setShowSources] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopyAnswer = () => {
    if (result?.answer) {
      navigator.clipboard.writeText(result.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div
      className={`bg-white rounded-xl border ${conf.accentBorder} p-5 shadow-sm hover:shadow-md transition-all duration-200 flex flex-col justify-between space-y-4 group`}
    >
      <div className="space-y-3">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full transition-transform duration-200 group-hover:scale-125" style={{ backgroundColor: conf.color }} />
            <h4 className="font-bold text-sm text-slate-900">{conf.name}</h4>
          </div>
          <div className="flex items-center gap-2">
            {result.exit_type && (
              <span
                className={`text-[10px] px-2.5 py-0.5 rounded-full font-mono font-bold border shadow-2xs transition-transform duration-200 group-hover:scale-105 ${
                  result.exit_type === "ANSWER"
                    ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                    : result.exit_type === "PARTIAL"
                    ? "bg-amber-50 text-amber-800 border-amber-200"
                    : "bg-rose-50 text-rose-800 border-rose-200"
                }`}
              >
                {result.exit_type}
              </span>
            )}
            <button
              onClick={handleCopyAnswer}
              className="p-1 rounded hover:bg-slate-100 transition-colors cursor-pointer active:scale-90"
              title="Copy answer"
            >
              {copied ? (
                <span className="inline-flex items-center gap-1 text-[10px] text-emerald-600 font-mono font-bold bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200 spell-pop">
                  <Icons.Check /> Copied
                </span>
              ) : (
                <Icons.Copy />
              )}
            </button>
          </div>
        </div>

        <div className="text-xs sm:text-sm text-slate-800 leading-relaxed font-sans min-h-[72px] whitespace-pre-wrap">
          {result.answer || <span className="text-slate-400 italic">No answer synthesized</span>}
        </div>
      </div>

      <div className="pt-3 border-t border-slate-100 space-y-2.5">
        <div className="flex items-center justify-between text-[11px] font-mono font-bold text-slate-600">
          <span>🪙 {result.tokens_used ?? 0} tokens</span>
          <span>⏱ {result.latency_ms ?? 0}ms</span>
        </div>

        {Array.isArray(result.sources) && result.sources.length > 0 && (
          <div className="pt-1">
            <button
              onClick={() => setShowSources(!showSources)}
              className="text-[11px] text-orange-600 hover:text-orange-700 font-bold flex items-center gap-1 cursor-pointer"
            >
              {showSources ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
              <span>{showSources ? "Hide" : "View"} Cited Sources ({result.sources.length})</span>
            </button>
            {showSources && (
              <div className="mt-2 p-2.5 bg-slate-50 rounded-lg border border-slate-200 text-[10px] text-slate-600 font-mono space-y-1 max-h-32 overflow-y-auto">
                {result.sources.map((src: string, i: number) => (
                  <div key={i} className="truncate">• {src}</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
