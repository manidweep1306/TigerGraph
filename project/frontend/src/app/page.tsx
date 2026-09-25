"use client";

import React, { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";
import { TabId, BenchmarkRecord, SummaryStats, HealthResponse } from "@/types";
import { Sidebar } from "@/components/Sidebar";
import { DashboardHeader, DashboardFooter } from "@/components/DashboardLayout";
import { StatsSummaryBar } from "@/components/StatsSummaryBar";
import { BenchmarkTable } from "@/components/BenchmarkTable";
import { TokenMetrics } from "@/components/TokenMetrics";
import { AccuracyPanel } from "@/components/AccuracyPanel";
import { AgentTracePanel } from "@/components/AgentTracePanel";
import { ConfusionMatrixPanel } from "@/components/ConfusionMatrixPanel";
import { QueryPlayground } from "@/components/QueryPlayground";

interface Notification {
  text: string;
  type: "success" | "error" | "info";
}

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabId>("benchmark");
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkRecord[]>([]);
  const [stats, setStats] = useState<SummaryStats | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionMessage, setActionMessage] = useState<Notification | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const res = await api.health();
      setHealthStatus(res);
    } catch {
      setHealthStatus({ status: "error", baseline_fairness: "FAIL" });
    }
  }, []);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [results, summary] = await Promise.all([
        api.getBenchmarkResults(),
        api.getSummaryStats(),
      ]);
      setBenchmarkData(Array.isArray(results) ? results : []);
      setStats(summary && summary.total_questions ? summary : null);
    } catch (e) {
      console.error("Failed to load dashboard data:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    loadData();
  }, [checkHealth, loadData]);

  const handleTriggerBenchmark = async (limit?: number) => {
    try {
      await api.runBenchmark(limit);
      setActionMessage({
        text: `Benchmark execution initiated in background (${limit ? `${limit} questions` : "full suite"}). Syncing...`,
        type: "info",
      });
      setTimeout(() => {
        loadData();
        setActionMessage(null);
      }, 3000);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "An unexpected error occurred";
      setActionMessage({ text: `Failed to trigger benchmark: ${message}`, type: "error" });
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 flex flex-col md:flex-row font-sans selection:bg-orange-100 selection:text-orange-900">
      {/* ── Left Vertical Navigation Sidebar ── */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        benchmarkCount={benchmarkData.length}
      />

      {/* ── Main Viewport Content Area ── */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#F8FAFC]">
        {/* Top Header Bar */}
        <DashboardHeader
          activeTab={activeTab}
          healthStatus={healthStatus}
          loading={loading}
          onRunSample={handleTriggerBenchmark}
          onRefresh={loadData}
        />

        {/* Global Summary KPI Bar */}
        {stats && <StatsSummaryBar stats={stats} />}

        {/* Notification Toast */}
        {actionMessage && (
          <div className="mx-6 mt-4">
            <div
              className={`p-3 rounded-xl border text-xs flex items-center justify-between shadow-sm transition-all spell-pop ${
                actionMessage.type === "error"
                  ? "bg-rose-50 border-rose-200 text-rose-800"
                  : actionMessage.type === "info"
                  ? "bg-blue-50 border-blue-200 text-blue-800"
                  : "bg-emerald-50 border-emerald-200 text-emerald-800"
              }`}
            >
              <span className="font-semibold">{actionMessage.text}</span>
              <button
                onClick={() => setActionMessage(null)}
                className="text-slate-400 hover:text-slate-700 font-bold px-1.5 py-0.5 cursor-pointer"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Main Content Viewport */}
        <main className="flex-1 p-6 max-w-7xl w-full mx-auto">
          {activeTab === "benchmark" && (
            <BenchmarkTable
              data={benchmarkData}
              loading={loading}
              onTriggerRun={handleTriggerBenchmark}
            />
          )}
          {activeTab === "tokens" && <TokenMetrics data={benchmarkData} stats={stats} />}
          {activeTab === "accuracy" && <AccuracyPanel data={benchmarkData} stats={stats} />}
          {activeTab === "trace" && <AgentTracePanel />}
          {activeTab === "matrix" && <ConfusionMatrixPanel data={benchmarkData} />}
          {activeTab === "playground" && <QueryPlayground />}
        </main>

        {/* Bottom App Footer */}
        <DashboardFooter />
      </div>
    </div>
  );
}
