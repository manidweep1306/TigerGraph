import React from "react";
import { BenchmarkRecord, SummaryStats } from "@/types";
import { PIPELINE_CONFIG } from "@/lib/constants";
import { EmptyState, AccuracyPill, avg } from "./Common";

function computeAccuracyRate(records: BenchmarkRecord[], pipeline: "rag" | "graphrag" | "agentic"): number {
  const scored = records.filter((r) => r[pipeline]?.accuracy_score);
  if (!scored.length) return 0;
  return scored.filter((r) => r[pipeline]?.accuracy_score === "PASS").length / scored.length;
}

export function AccuracyPanel({ data }: { data: BenchmarkRecord[]; stats: SummaryStats | null }) {
  if (!data.length) {
    return <EmptyState message="No benchmark data available. Run benchmark to populate accuracy analysis." />;
  }

  const qtypes = [...new Set(data.map((r) => r.qtype))].sort();

  return (
    <div className="space-y-6">
      {/* Accuracy Matrix Table */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div>
          <h3 className="font-bold text-sm text-slate-900">Pass Rate by Question Type</h3>
          <p className="text-xs text-slate-500 mt-1">
            Exact match / semantic validation accuracy across reasoning question categories.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-600 uppercase font-bold border-b border-slate-200">
              <tr>
                <th className="py-3 px-4">Question Category</th>
                <th className="py-3 px-4 text-center">Standard RAG</th>
                <th className="py-3 px-4 text-center">GraphRAG</th>
                <th className="py-3 px-4 text-center">Agentic VoI</th>
                <th className="py-3 px-4 text-center">Δ (Agentic vs RAG)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-sans">
              {qtypes.map((qt) => {
                const subset = data.filter((r) => r.qtype === qt);
                const ragAcc = computeAccuracyRate(subset, "rag");
                const gragAcc = computeAccuracyRate(subset, "graphrag");
                const agAcc = computeAccuracyRate(subset, "agentic");
                const delta = agAcc - ragAcc;
                return (
                  <tr key={qt} className="hover:bg-slate-50 transition">
                    <td className="py-3 px-4 font-bold text-slate-800">
                      <span className="capitalize">{qt.replace("_", " ")}</span>
                      <span className="ml-2 text-[10px] text-slate-400 font-mono font-normal">({subset.length} q)</span>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <AccuracyPill value={ragAcc} />
                    </td>
                    <td className="py-3 px-4 text-center">
                      <AccuracyPill value={gragAcc} />
                    </td>
                    <td className="py-3 px-4 text-center">
                      <AccuracyPill value={agAcc} />
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span
                        className={`font-mono font-bold text-xs px-2.5 py-0.5 rounded border ${
                          delta > 0
                            ? "text-emerald-800 bg-emerald-50 border-emerald-200"
                            : delta < 0
                            ? "text-rose-800 bg-rose-50 border-rose-200"
                            : "text-slate-500 bg-slate-100 border-slate-200"
                        }`}
                      >
                        {delta > 0 ? "+" : ""}
                        {(delta * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* BERTScore Overview Cards */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div>
          <h3 className="font-bold text-sm text-slate-900">Semantic BERTScore F1 Evaluation</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Measures semantic overlap against ground truth answers using contextual embeddings.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          {(["rag", "graphrag", "agentic"] as const).map((p) => {
            const conf = PIPELINE_CONFIG[p];
            const scores = data.map((r) => r[p]?.bertscore_f1 ?? 0).filter((v) => v > 0);
            const avgScore = scores.length ? avg(scores) : 0;
            return (
              <div
                key={p}
                className="bg-slate-50 rounded-xl p-5 border border-slate-200 flex flex-col items-center justify-center text-center space-y-1 shadow-2xs"
              >
                <span className="text-3xl font-extrabold text-slate-900 font-mono tracking-tight">
                  {avgScore.toFixed(3)}
                </span>
                <span className="text-xs font-bold text-slate-700 flex items-center gap-1.5 mt-1">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: conf.color }} />
                  {conf.name}
                </span>
                <span className="text-[11px] text-slate-400 font-mono">Mean BERTScore F1</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
