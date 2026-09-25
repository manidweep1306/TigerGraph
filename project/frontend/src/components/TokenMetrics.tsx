import React from "react";
import { BenchmarkRecord, SummaryStats } from "@/types";
import { PIPELINE_CONFIG } from "@/lib/constants";
import { EmptyState, MetricRow, avg } from "./Common";

export function TokenMetrics({ data, stats }: { data: BenchmarkRecord[]; stats: SummaryStats | null }) {
  if (!data.length) {
    return <EmptyState message="No benchmark data available. Run benchmark to populate token metrics." />;
  }

  const qtypes = [...new Set(data.map((r) => r.qtype))].sort();

  const avgTokensByType = qtypes.map((qt) => {
    const subset = data.filter((r) => r.qtype === qt);
    return {
      qtype: qt,
      rag: avg(subset.map((r) => r.rag?.tokens_used ?? 0)),
      graphrag: avg(subset.map((r) => r.graphrag?.tokens_used ?? 0)),
      agentic: avg(subset.map((r) => r.agentic?.tokens_used ?? 0)),
    };
  });

  const maxTokens = Math.max(1, ...avgTokensByType.flatMap((d) => [d.rag, d.graphrag, d.agentic]));

  return (
    <div className="space-y-6">
      {/* 3 Overview Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {(["rag", "graphrag", "agentic"] as const).map((key) => {
          const conf = PIPELINE_CONFIG[key];
          const s = stats?.[key];
          return (
            <div
              key={key}
              className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs relative overflow-hidden"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: conf.color }} />
                  <h3 className="font-bold text-sm text-slate-900">{conf.name}</h3>
                </div>
                <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700 font-mono font-semibold border border-slate-200">
                  {conf.tag}
                </span>
              </div>

              <div className="space-y-3">
                <MetricRow label="Average Tokens" value={`${Math.round(s?.avg_tokens ?? 0)}`} unit="tok" />
                <MetricRow label="Average Latency" value={`${Math.round(s?.avg_latency_ms ?? 0)}`} unit="ms" />
                <MetricRow label="Accuracy Pass Rate" value={`${((s?.accuracy ?? 0) * 100).toFixed(1)}`} unit="%" />
                <MetricRow label="Avg BERTScore F1" value={`${(s?.avg_bertscore ?? 0).toFixed(3)}`} unit="" />
              </div>
            </div>
          );
        })}
      </div>

      {/* Horizontal Bar Chart by Question Type */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-6">
        <div>
          <h3 className="font-bold text-sm text-slate-900">Average Token Consumption by Question Complexity</h3>
          <p className="text-xs text-slate-500 mt-1">
            Measures how many tokens each retrieval strategy consumes across different reasoning categories.
          </p>
        </div>

        <div className="space-y-5">
          {avgTokensByType.map((row) => (
            <div key={row.qtype} className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-slate-800 uppercase tracking-wide">
                  {row.qtype.replace("_", " ")}
                </span>
              </div>
              <div className="space-y-2">
                {(["rag", "graphrag", "agentic"] as const).map((key) => {
                  const val = row[key];
                  const pct = Math.max(4, (val / maxTokens) * 100);
                  const conf = PIPELINE_CONFIG[key];
                  return (
                    <div key={key} className="flex items-center gap-3">
                      <span className="text-xs text-slate-600 w-24 text-right truncate font-semibold">
                        {conf.name}
                      </span>
                      <div className="flex-1 bg-slate-100 rounded-full h-5 relative overflow-hidden border border-slate-200">
                        <div
                          className="h-full rounded-full transition-all duration-700 flex items-center justify-end pr-2.5"
                          style={{ width: `${pct}%`, backgroundColor: conf.color }}
                        >
                          <span className="text-[10px] text-white font-mono font-bold drop-shadow">
                            {Math.round(val)} tok
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Token Distribution Histogram */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs space-y-4">
        <div>
          <h3 className="font-bold text-sm text-slate-900">Token Cost Distribution Across Queries</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Query count distribution per token consumption tier.
          </p>
        </div>
        <TokenHistogram data={data} />
      </div>
    </div>
  );
}

function TokenHistogram({ data }: { data: BenchmarkRecord[] }) {
  const buckets = [0, 200, 500, 1000, 2000, 5000, Infinity];
  const labels = ["0-200", "200-500", "500-1k", "1k-2k", "2k-5k", "5k+"];

  const bucketCounts = labels.map((_, i) =>
    Object.fromEntries(
      (["rag", "graphrag", "agentic"] as const).map((key) => [
        key,
        data.filter((r) => {
          const tok = r[key]?.tokens_used ?? 0;
          return tok >= buckets[i] && tok < buckets[i + 1];
        }).length,
      ])
    )
  );

  const maxCount = Math.max(1, ...bucketCounts.flatMap((b) => Object.values(b) as number[]));

  return (
    <div className="space-y-3 pt-2">
      <div className="flex items-end gap-3 h-40 pt-4">
        {labels.map((label, i) => (
          <div key={label} className="flex-1 flex flex-col items-center gap-1.5 h-full">
            <div className="w-full flex items-end gap-1 flex-1 pb-1">
              {(["rag", "graphrag", "agentic"] as const).map((key) => {
                const count = (bucketCounts[i][key] as number) || 0;
                const height = Math.max(4, (count / maxCount) * 100);
                const conf = PIPELINE_CONFIG[key];
                return (
                  <div
                    key={key}
                    className="flex-1 rounded-t-sm transition-all relative group"
                    style={{ height: `${height}%`, backgroundColor: conf.color }}
                  >
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute -top-7 left-1/2 -translate-x-1/2 px-2 py-0.5 bg-slate-800 text-white text-[10px] font-mono rounded pointer-events-none whitespace-nowrap z-10 border border-slate-700 shadow-md">
                      {conf.name}: {count}
                    </div>
                  </div>
                );
              })}
            </div>
            <span className="text-[11px] text-slate-500 font-mono font-semibold">{label}</span>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-center gap-6 pt-2 text-xs text-slate-600 font-semibold">
        {(["rag", "graphrag", "agentic"] as const).map((key) => (
          <div key={key} className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: PIPELINE_CONFIG[key].color }} />
            <span>{PIPELINE_CONFIG[key].name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
