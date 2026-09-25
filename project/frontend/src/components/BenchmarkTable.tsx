import React, { useState, useEffect, useMemo } from "react";
import { BenchmarkRecord } from "@/types";
import { PIPELINE_CONFIG } from "@/lib/constants";
import { Icons } from "./Icons";
import { LoadingSpinner } from "./Common";

export function BenchmarkTable({
  data,
  loading,
  onTriggerRun,
}: {
  data: BenchmarkRecord[];
  loading: boolean;
  onTriggerRun: (limit?: number) => void;
}) {
  const [filterType, setFilterType] = useState("all");
  const [filterResult, setFilterResult] = useState<"all" | "agentic_win" | "failures">("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const pageSize = 20;

  const qtypes = ["all", "lookup", "temporal", "multi_hop", "aggregation", "superlative"];

  const filtered = useMemo(() => {
    return data.filter((r) => {
      if (filterType !== "all" && r.qtype !== filterType) return false;
      if (
        search &&
        !r.question.toLowerCase().includes(search.toLowerCase()) &&
        !r.question_id.toLowerCase().includes(search.toLowerCase())
      ) {
        return false;
      }
      if (filterResult === "agentic_win") {
        return r.agentic?.accuracy_score === "PASS" && r.rag?.accuracy_score === "FAIL";
      }
      if (filterResult === "failures") {
        return (
          r.rag?.accuracy_score === "FAIL" ||
          r.graphrag?.accuracy_score === "FAIL" ||
          r.agentic?.accuracy_score === "FAIL"
        );
      }
      return true;
    });
  }, [data, filterType, filterResult, search]);

  const totalPages = Math.ceil(filtered.length / pageSize) || 1;
  const paginatedRows = useMemo(() => {
    const start = (page - 1) * pageSize;
    return filtered.slice(start, start + pageSize);
  }, [filtered, page]);

  useEffect(() => {
    setPage(1);
  }, [filterType, filterResult, search]);

  if (loading) return <LoadingSpinner label="Fetching benchmark dataset results..." />;

  return (
    <div className="space-y-4">
      {/* Search & Control Filter Bar */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3.5">
        <div className="flex flex-wrap gap-3.5 items-center justify-between">
          <div className="flex-1 min-w-[280px] relative">
            <span className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
              <Icons.Search />
            </span>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by question text or ID (e.g., q_10)..."
              className="w-full pl-9 pr-8 py-2 bg-slate-50 border border-slate-300 rounded-lg text-xs sm:text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:border-[#FF7A00] focus:ring-1 focus:ring-[#FF7A00] transition"
            />
            {search && (
              <button
                onClick={() => setSearch("")}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-700 text-xs font-bold cursor-pointer"
              >
                ✕
              </button>
            )}
          </div>

          {/* Result Filters */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-500 font-semibold mr-1">Filter View:</span>
            <button
              onClick={() => setFilterResult("all")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition border cursor-pointer ${
                filterResult === "all"
                  ? "bg-slate-900 text-white border-slate-900 shadow-xs"
                  : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
              }`}
            >
              All Records
            </button>
            <button
              onClick={() => setFilterResult("agentic_win")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition border cursor-pointer ${
                filterResult === "agentic_win"
                  ? "bg-[#FFF2E8] text-[#D4380D] border-[#FFD591] font-bold shadow-xs"
                  : "bg-white text-[#D4380D] border-orange-200 hover:bg-orange-50"
              }`}
              title="Questions where Agentic passed while Standard RAG failed"
            >
              ⚡ Agentic Wins
            </button>
            <button
              onClick={() => setFilterResult("failures")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition border cursor-pointer ${
                filterResult === "failures"
                  ? "bg-rose-50 text-rose-700 border-rose-200 font-bold shadow-xs"
                  : "bg-white text-rose-700 border-rose-200 hover:bg-rose-50"
              }`}
            >
              ⚠️ Any Failure
            </button>
          </div>
        </div>

        {/* Question Type Filter Pills (TigerGraph Pill Badges) */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-500 font-semibold mr-1">Q-Type:</span>
            {qtypes.map((qt) => {
              const isActive = filterType === qt;
              return (
                <button
                  key={qt}
                  onClick={() => setFilterType(qt)}
                  className={`px-3 py-1 text-xs rounded-full font-semibold transition capitalize border cursor-pointer ${
                    isActive
                      ? "bg-[#FF7A00] text-white border-[#FF7A00] shadow-xs"
                      : "bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200"
                  }`}
                >
                  {qt.replace("_", " ")}
                </button>
              );
            })}
          </div>

          <div className="text-xs text-slate-500 font-mono">
            Showing <strong className="text-slate-900 font-bold">{filtered.length}</strong> of{" "}
            <span className="text-slate-700 font-semibold">{data.length}</span> questions
          </div>
        </div>
      </div>

      {/* Benchmark Table / Empty State */}
      {filtered.length === 0 ? (
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center space-y-4 shadow-sm">
          <div className="w-14 h-14 mx-auto rounded-full bg-orange-50 border border-orange-200 flex items-center justify-center text-orange-600 text-2xl">
            <Icons.Benchmark />
          </div>
          <h3 className="text-base font-bold text-slate-900">No Benchmark Records Available</h3>
          <p className="text-xs sm:text-sm text-slate-500 max-w-md mx-auto leading-relaxed">
            No evaluation results were found matching your filter criteria, or the benchmark evaluation suite has
            not been executed yet.
          </p>
          <div className="pt-2">
            <button
              onClick={() => onTriggerRun(20)}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#FF7A00] hover:bg-[#E66E00] text-white text-xs font-bold rounded-lg shadow-sm transition cursor-pointer"
            >
              <Icons.Play /> Run 20 Questions Benchmark
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-50 text-slate-600 uppercase tracking-wider font-bold border-b border-slate-200">
                <tr>
                  <th className="px-4 py-3.5 w-20">Q-ID</th>
                  <th className="px-4 py-3.5">Question Text</th>
                  <th className="px-3 py-3.5 text-center w-28">Type</th>
                  <th className="px-3 py-3.5 text-center w-24">RAG</th>
                  <th className="px-3 py-3.5 text-center w-24">GraphRAG</th>
                  <th className="px-3 py-3.5 text-center w-24">Agentic</th>
                  <th className="px-4 py-3.5 text-center w-40">Tokens (R / G / A)</th>
                  <th className="px-3 py-3.5 text-center w-12">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-sans">
                {paginatedRows.map((row) => (
                  <BenchmarkRow key={row.question_id} row={row} />
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="px-3 py-1.5 rounded-lg bg-white border border-slate-300 hover:bg-slate-50 font-semibold disabled:opacity-40 transition cursor-pointer"
              >
                ← Previous
              </button>
              <span className="font-medium">
                Page <strong className="text-slate-900 font-bold">{page}</strong> of{" "}
                <strong className="text-slate-900 font-bold">{totalPages}</strong>
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="px-3 py-1.5 rounded-lg bg-white border border-slate-300 hover:bg-slate-50 font-semibold disabled:opacity-40 transition cursor-pointer"
              >
                Next →
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function BenchmarkRow({ row }: { row: BenchmarkRecord }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(row.question);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderBadge = (score?: string) => {
    if (score === "PASS") {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-800 border border-emerald-200 text-[11px] font-mono font-bold">
          <Icons.Check /> PASS
        </span>
      );
    }
    if (score === "FAIL") {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-rose-50 text-rose-800 border border-rose-200 text-[11px] font-mono font-bold">
          <Icons.XIcon /> FAIL
        </span>
      );
    }
    return <span className="text-slate-400 font-mono font-bold">—</span>;
  };

  const ragTok = row.rag?.tokens_used ?? 0;
  const gragTok = row.graphrag?.tokens_used ?? 0;
  const agTok = row.agentic?.tokens_used ?? 0;

  return (
    <>
      <tr
        onClick={() => setExpanded(!expanded)}
        className="hover:bg-slate-50/80 cursor-pointer transition select-none group"
      >
        <td className="px-4 py-3 font-mono font-bold text-slate-700 group-hover:text-orange-600">
          {row.question_id}
        </td>
        <td className="px-4 py-3 text-slate-800 font-medium max-w-md truncate" title={row.question}>
          <div className="flex items-center gap-2">
            <span className="truncate">{row.question}</span>
            <button
              onClick={handleCopy}
              className="opacity-0 group-hover:opacity-100 transition-all p-1 rounded hover:bg-slate-200/80 active:scale-90 cursor-pointer"
              title="Copy question text"
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
        </td>
        <td className="px-3 py-3 text-center">
          <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 text-[10px] font-bold uppercase border border-slate-200">
            {row.qtype}
          </span>
        </td>
        <td className="px-3 py-3 text-center">{renderBadge(row.rag?.accuracy_score)}</td>
        <td className="px-3 py-3 text-center">{renderBadge(row.graphrag?.accuracy_score)}</td>
        <td className="px-3 py-3 text-center">{renderBadge(row.agentic?.accuracy_score)}</td>
        <td className="px-4 py-3 text-center font-mono text-[11px] font-medium">
          <span className="text-blue-700 font-bold">{ragTok}</span> /{" "}
          <span className="text-purple-700 font-bold">{gragTok}</span> /{" "}
          <span className="text-orange-600 font-bold">{agTok}</span>
        </td>
        <td className="px-3 py-3 text-center text-slate-400">
          {expanded ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
        </td>
      </tr>

      {/* Expanded Breakdown */}
      {expanded && (
        <tr className="bg-slate-50 border-b border-slate-200">
          <td colSpan={8} className="px-6 py-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {(["rag", "graphrag", "agentic"] as const).map((p) => {
                const conf = PIPELINE_CONFIG[p];
                const res = row[p];
                return (
                  <div
                    key={p}
                    className={`bg-white rounded-xl p-4 border ${conf.accentBorder} space-y-2.5 shadow-xs`}
                  >
                    <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                      <span className="font-bold text-xs text-slate-800 uppercase flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: conf.color }} />
                        {conf.name}
                      </span>
                      {res?.exit_type && (
                        <span className="text-[10px] px-2 py-0.5 rounded font-mono font-bold bg-slate-100 text-slate-700 border border-slate-200">
                          {res.exit_type}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-700 leading-relaxed min-h-[48px] font-sans">
                      {res?.answer || <span className="text-slate-400 italic">No output recorded</span>}
                    </p>
                    <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] font-mono text-slate-500">
                      <span>BERTScore: <strong className="text-slate-800">{(res?.bertscore_f1 ?? 0).toFixed(3)}</strong></span>
                      <span>Tokens: <strong className="text-slate-800">{res?.tokens_used ?? 0}</strong></span>
                      <span>Latency: <strong className="text-slate-800">{res?.latency_ms ?? 0}ms</strong></span>
                    </div>
                  </div>
                );
              })}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
