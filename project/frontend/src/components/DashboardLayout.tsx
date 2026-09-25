import React from "react";
import { TabId, HealthResponse } from "@/types";
import { Icons } from "./Icons";

interface HeaderProps {
  activeTab: TabId;
  healthStatus: HealthResponse | null;
  loading: boolean;
  onRunSample: (limit?: number) => void;
  onRefresh: () => void;
}

const TAB_TITLES: Record<TabId, string> = {
  benchmark: "Benchmark Explorer",
  tokens: "Token Efficiency",
  accuracy: "Accuracy & BERTScore",
  trace: "Investigation Trace",
  matrix: "Routing Matrix",
  playground: "Live Query Bench",
};

export function DashboardHeader({
  activeTab,
  healthStatus,
  loading,
  onRunSample,
  onRefresh,
}: HeaderProps) {
  return (
    <header className="bg-white border-b border-slate-200 px-6 py-[16px] min-h-[60px] sticky top-0 z-10 flex flex-wrap items-center justify-between gap-4 shadow-xs">
      <div className="flex items-center gap-3">
        <h2 className="text-base font-bold text-slate-900">{TAB_TITLES[activeTab]}</h2>
      </div>

      <div className="flex items-center gap-3">
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-mono font-medium shadow-xs ${
            healthStatus?.baseline_fairness === "PASS"
              ? "bg-emerald-50 text-emerald-800 border-emerald-200"
              : healthStatus?.baseline_fairness === "FAIL"
              ? "bg-rose-50 text-rose-800 border-rose-200"
              : "bg-slate-100 text-slate-600 border-slate-200"
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              healthStatus?.baseline_fairness === "PASS"
                ? "bg-emerald-500"
                : healthStatus?.baseline_fairness === "FAIL"
                ? "bg-rose-500"
                : "bg-slate-400"
            }`}
          />
          <span className="font-sans text-slate-500 text-[11px]">Fairness:</span>
          <span className="font-bold">{healthStatus?.baseline_fairness || "CHECKING"}</span>
        </div>

        <button
          onClick={() => onRunSample(10)}
          disabled={loading}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 bg-white hover:bg-slate-50 active:scale-95 text-slate-700 text-xs font-semibold rounded-lg border border-slate-300 transition-all duration-200 shadow-xs hover:shadow hover:border-slate-400 cursor-pointer disabled:opacity-50"
        >
          <Icons.Play /> Run Sample (10)
        </button>

        <button
          onClick={onRefresh}
          disabled={loading}
          className="spell-shimmer-btn inline-flex items-center gap-1.5 px-4 py-1.5 bg-[#FF7A00] hover:bg-[#E66E00] active:scale-95 text-white text-xs font-bold rounded-lg shadow-sm hover:shadow-md hover:shadow-orange-500/20 transition-all duration-200 cursor-pointer disabled:opacity-50"
        >
          <Icons.Refresh className={loading ? "animate-spin" : "transition-transform duration-300 group-hover:rotate-180"} />
          <span>{loading ? "Syncing..." : "Refresh"}</span>
        </button>
      </div>
    </header>
  );
}

export function DashboardFooter() {
  return (
    <footer className="border-t border-slate-200 bg-white py-[14px] px-6 min-h-[48px] flex items-center text-[13px] text-slate-600">
      <div className="max-w-7xl w-full mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
        <span className="font-medium text-slate-700">
          TigerGraph Savanna Hackathon · Olympic Events Benchmark · Value-of-Information (VoI) Orchestrator
        </span>
        <span className="font-mono text-[12px] font-semibold text-slate-700 bg-slate-100 px-2.5 py-0.5 rounded-md border border-slate-200">
          Next.js 14 · LangGraph · Gemini 3.6 Flash · Savanna Cloud
        </span>
      </div>
    </footer>
  );
}
