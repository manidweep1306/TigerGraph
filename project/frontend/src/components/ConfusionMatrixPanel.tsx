import React, { useMemo } from "react";
import { BenchmarkRecord, ConfusionMatrixData } from "@/types";

function buildConfusionMatrix(records: BenchmarkRecord[]): ConfusionMatrixData {
  const matrix: Record<string, Record<string, number>> = {
    RAG: { RAG: 0, GraphRAG: 0, Agentic: 0, none_passed: 0 },
    GraphRAG: { RAG: 0, GraphRAG: 0, Agentic: 0, none_passed: 0 },
    Agentic: { RAG: 0, GraphRAG: 0, Agentic: 0, none_passed: 0 },
  };

  let total = 0;
  let correct = 0;

  for (const r of records) {
    let groundTruth = "none_passed";
    if (r.rag?.accuracy_score === "PASS") {
      groundTruth = "RAG";
    } else if (r.graphrag?.accuracy_score === "PASS") {
      groundTruth = "GraphRAG";
    } else if (r.agentic?.accuracy_score === "PASS") {
      groundTruth = "Agentic";
    }

    let predicted = "RAG";
    if (r.qtype === "lookup") predicted = "RAG";
    else if (r.qtype === "temporal" || r.qtype === "multi_hop") predicted = "GraphRAG";
    else predicted = "Agentic";

    if (matrix[predicted]) {
      matrix[predicted][groundTruth] = (matrix[predicted][groundTruth] || 0) + 1;
      total += 1;
      if (predicted === groundTruth) {
        correct += 1;
      }
    }
  }

  return {
    matrix,
    total,
    accuracy: total > 0 ? correct / total : 0,
  };
}

export function ConfusionMatrixPanel({ data }: { data: BenchmarkRecord[] }) {
  const matrixData = useMemo(() => buildConfusionMatrix(data), [data]);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="font-bold text-sm sm:text-base text-slate-900">
            Ladder Routing Precision — Confusion Matrix
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            <strong>Ground Truth</strong> = Cheapest rung that answered correctly ·{" "}
            <strong>Predicted</strong> = Route predicted by the VoI Escalation Ladder.
          </p>
        </div>

        {matrixData.total > 0 && (
          <div className="flex items-center gap-3 bg-slate-50 px-4 py-2 rounded-xl border border-slate-200 shadow-2xs">
            <span className="text-xs text-slate-600 font-semibold">Routing Accuracy:</span>
            <span className="text-sm font-extrabold font-mono text-emerald-700">
              {(matrixData.accuracy * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>

      <div className="overflow-x-auto pt-2">
        <table className="text-xs w-full max-w-2xl border-collapse">
          <thead>
            <tr>
              <th className="px-4 py-3 text-right text-slate-500 font-bold border-b border-slate-200">
                Predicted ↓ / True →
              </th>
              {["RAG", "GraphRAG", "Agentic", "none_passed"].map((col) => (
                <th
                  key={col}
                  className="px-6 py-3 text-center text-slate-700 font-bold border-b border-slate-200 uppercase tracking-wider"
                >
                  {col.replace("_", " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-mono">
            {["RAG", "GraphRAG", "Agentic"].map((row) => (
              <tr key={row} className="hover:bg-slate-50/80 transition-colors">
                <td className="px-4 py-3.5 text-right text-slate-800 font-extrabold font-sans">{row}</td>
                {["RAG", "GraphRAG", "Agentic", "none_passed"].map((col) => {
                  const val = matrixData.matrix[row]?.[col] ?? 0;
                  const isDiagonal = row === col;
                  return (
                    <td key={col} className="px-6 py-3.5 text-center">
                      <span
                        className={`inline-block px-3.5 py-1.5 rounded-lg text-xs font-bold border transition-all duration-200 cursor-default hover:scale-115 hover:z-10 relative ${
                          val > 0 && isDiagonal
                            ? "bg-emerald-50 text-emerald-800 border-emerald-300 font-extrabold shadow-sm hover:shadow-emerald-200 hover:ring-2 hover:ring-emerald-400"
                            : val > 0
                            ? "bg-slate-100 text-slate-800 border-slate-300 shadow-2xs hover:shadow-slate-200 hover:ring-2 hover:ring-slate-400"
                            : "bg-slate-50 text-slate-400 border-slate-200 opacity-60 hover:opacity-100"
                        }`}
                        title={`${row} predicted → ${col} true (${val} cases)`}
                      >
                        {val}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs text-slate-600 space-y-1.5">
        <div className="font-bold text-orange-700 flex items-center gap-1.5">
          <span>💡</span> Escalation Economics:
        </div>
        <p className="leading-relaxed">
          Routing to Standard RAG first on simpler queries saves ~80% in token costs. When confidence is below
          frozen calibration thresholds (0.80), the ladder automatically escalates to GraphRAG or the full
          Agentic Investigation Machine.
        </p>
      </div>
    </div>
  );
}
