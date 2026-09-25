import React from "react";
import { SummaryStats } from "@/types";
import { PIPELINE_CONFIG } from "@/lib/constants";

export function StatsSummaryBar({ stats }: { stats: SummaryStats }) {
  const pipelines = [PIPELINE_CONFIG.rag, PIPELINE_CONFIG.graphrag, PIPELINE_CONFIG.agentic];

  return (
    <div className="border-b border-slate-200 bg-white px-6 py-2.5 shadow-2xs">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-5">
          {pipelines.map((p) => {
            const s = stats[p.id as keyof SummaryStats] as { count?: number; accuracy?: number; avg_tokens?: number; avg_latency_ms?: number } | undefined;
            if (!s || s.count === undefined) return null;
            const acc = ((s.accuracy ?? 0) * 100).toFixed(1);
            return (
              <div key={p.id} className="flex items-center gap-2 px-2.5 py-1 rounded-lg hover:bg-slate-50 transition-all duration-200 cursor-default group">
                <span className="w-2.5 h-2.5 rounded-full transition-transform duration-200 group-hover:scale-125" style={{ backgroundColor: p.color }} />
                <span className="text-xs font-bold text-slate-800">{p.name}</span>
                <span className={`text-xs font-bold font-mono px-2 py-0.5 rounded-md border ${p.badgeClass} transition-transform duration-200 group-hover:scale-105 shadow-2xs`}>
                  {acc}% Acc
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  {Math.round(s.avg_tokens ?? 0)} tok · {Math.round(s.avg_latency_ms ?? 0)}ms
                </span>
              </div>
            );
          })}
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-600 bg-slate-100 hover:bg-slate-200/80 px-2.5 py-1 rounded-md border border-slate-200 transition-colors shadow-2xs">
          <span className="text-slate-900 font-bold">{stats.total_questions}</span>
          <span>evaluated questions</span>
        </div>
      </div>
    </div>
  );
}
