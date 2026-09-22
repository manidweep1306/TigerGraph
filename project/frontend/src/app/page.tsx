"use client";

import React, { useState, useRef, useEffect } from "react";
import { api, BenchmarkRecord, SummaryStats } from "@/lib/api";

// ─── Main Dashboard Page ──────────────────────────────────────────────────────
export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<
    "benchmark" | "tokens" | "accuracy" | "trace" | "playground" | "matrix"
  >("benchmark");
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkRecord[]>([]);
  const [stats, setStats] = useState<SummaryStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState<string>("checking...");

  useEffect(() => {
    api.health().then((h) => setHealthStatus(h.baseline_fairness)).catch(() => setHealthStatus("ERROR"));
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [results, summary] = await Promise.all([
        api.getBenchmarkResults(),
        api.getSummaryStats(),
      ]);
      setBenchmarkData(results);
      setStats(summary);
    } catch (e) {
      console.error("Failed to load data:", e);
    }
    setLoading(false);
  };

  const TABS = [
    { id: "benchmark", label: "📊 Benchmark" },
    { id: "tokens", label: "🪙 Token Efficiency" },
    { id: "accuracy", label: "🎯 Accuracy" },
    { id: "trace", label: "🔍 Agent Trace" },
    { id: "matrix", label: "🗺️ Routing Matrix" },
    { id: "playground", label: "🧪 Live Query" },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-screen-2xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">🐯</span>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-orange-400 to-yellow-300 bg-clip-text text-transparent">
                Agentic GraphRAG Dashboard
              </h1>
              <p className="text-xs text-gray-400">TigerGraph Hackathon · Olympic Events Corpus</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span
              className={`text-xs px-3 py-1 rounded-full font-mono ${
                healthStatus === "PASS"
                  ? "bg-green-900 text-green-300"
                  : healthStatus === "checking..."
                  ? "bg-gray-800 text-gray-400"
                  : "bg-red-900 text-red-300"
              }`}
            >
              Baseline Fairness: {healthStatus}
            </span>
            <button
              onClick={loadData}
              disabled={loading}
              className="px-4 py-2 bg-orange-600 hover:bg-orange-500 text-white text-sm rounded-lg transition disabled:opacity-50"
            >
              {loading ? "Loading..." : "↻ Refresh"}
            </button>
          </div>
        </div>

        {/* Summary Stats Row */}
        {stats && <StatsSummaryBar stats={stats} />}

        {/* Tabs */}
        <div className="max-w-screen-2xl mx-auto px-6 flex gap-1 pb-0">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-2.5 text-sm font-medium rounded-t-lg transition ${
                activeTab === tab.id
                  ? "bg-gray-950 text-orange-400 border-b-2 border-orange-400"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </header>

      {/* Content */}
      <main className="max-w-screen-2xl mx-auto px-6 py-6">
        {activeTab === "benchmark" && <BenchmarkTable data={benchmarkData} loading={loading} />}
        {activeTab === "tokens" && <TokenMetrics data={benchmarkData} stats={stats} />}
        {activeTab === "accuracy" && <AccuracyPanel data={benchmarkData} stats={stats} />}
        {activeTab === "trace" && <AgentTracePanel />}
        {activeTab === "matrix" && <ConfusionMatrixPanel />}
        {activeTab === "playground" && <QueryPlayground />}
      </main>
    </div>
  );
}

// ─── Stats Summary Bar ────────────────────────────────────────────────────────
function StatsSummaryBar({ stats }: { stats: SummaryStats }) {
  const pipelines = [
    { key: "rag", label: "RAG", color: "blue" },
    { key: "graphrag", label: "GraphRAG", color: "purple" },
    { key: "agentic", label: "Agentic", color: "orange" },
  ];

  return (
    <div className="max-w-screen-2xl mx-auto px-6 py-3 flex gap-6 border-t border-gray-800">
      {pipelines.map(({ key, label, color }) => {
        const s = stats[key as keyof typeof stats] as any;
        if (!s?.count) return null;
        return (
          <div key={key} className="flex items-center gap-3">
            <div
              className={`w-2 h-2 rounded-full ${
                color === "blue" ? "bg-blue-400" : color === "purple" ? "bg-purple-400" : "bg-orange-400"
              }`}
            />
            <span className="text-xs text-gray-400">{label}</span>
            <span className="text-sm font-bold text-white">
              {(s.accuracy * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-gray-500">{Math.round(s.avg_tokens)} tok</span>
          </div>
        );
      })}
      <div className="ml-auto text-xs text-gray-500">
        {stats.total_questions} questions evaluated
      </div>
    </div>
  );
}

// ─── Benchmark Table ──────────────────────────────────────────────────────────
function BenchmarkTable({ data, loading }: { data: BenchmarkRecord[]; loading: boolean }) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<string>("question_id");

  const qtypes = ["all", "lookup", "temporal", "multi_hop", "aggregation", "superlative"];

  const filtered = data
    .filter((r) => filter === "all" || r.qtype === filter)
    .filter((r) => !search || r.question.toLowerCase().includes(search.toLowerCase()));

  if (loading) return <LoadingSpinner />;

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex gap-3 items-center flex-wrap">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search questions..."
          className="flex-1 min-w-60 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-orange-500"
        />
        <div className="flex gap-1">
          {qtypes.map((qt) => (
            <button
              key={qt}
              onClick={() => setFilter(qt)}
              className={`px-3 py-1.5 text-xs rounded-lg transition ${
                filter === qt
                  ? "bg-orange-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >
              {qt}
            </button>
          ))}
        </div>
        <span className="text-sm text-gray-500">{filtered.length} questions</span>
      </div>

      {/* Table */}
      {filtered.length === 0 ? (
        <EmptyState message="No benchmark results yet. Run /benchmark/run to generate results." />
      ) : (
        <div className="overflow-x-auto rounded-xl border border-gray-800">
          <table className="w-full text-sm">
            <thead className="bg-gray-900 text-gray-400 text-xs uppercase">
              <tr>
                <th className="px-4 py-3 text-left">ID</th>
                <th className="px-4 py-3 text-left">Question</th>
                <th className="px-4 py-3 text-center">Type</th>
                <th className="px-4 py-3 text-center">RAG</th>
                <th className="px-4 py-3 text-center">GraphRAG</th>
                <th className="px-4 py-3 text-center">Agentic</th>
                <th className="px-4 py-3 text-center">Tokens ↓</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {filtered.slice(0, 50).map((row) => (
                <BenchmarkRow key={row.question_id} row={row} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function BenchmarkRow({ row }: { row: BenchmarkRecord }) {
  const [expanded, setExpanded] = useState(false);

  const Badge = ({ score }: { score?: string }) =>
    score === "PASS" ? (
      <span className="px-2 py-0.5 bg-green-900 text-green-300 rounded text-xs font-mono">PASS</span>
    ) : score === "FAIL" ? (
      <span className="px-2 py-0.5 bg-red-900 text-red-300 rounded text-xs font-mono">FAIL</span>
    ) : (
      <span className="px-2 py-0.5 bg-gray-800 text-gray-500 rounded text-xs font-mono">—</span>
    );

  return (
    <>
      <tr
        className="hover:bg-gray-900/50 cursor-pointer transition"
        onClick={() => setExpanded(!expanded)}
      >
        <td className="px-4 py-3 font-mono text-xs text-gray-500">{row.question_id}</td>
        <td className="px-4 py-3 text-gray-200 max-w-xs truncate" title={row.question}>
          {row.question}
        </td>
        <td className="px-4 py-3 text-center">
          <span className="px-2 py-0.5 bg-gray-800 text-gray-400 rounded text-xs">
            {row.qtype}
          </span>
        </td>
        <td className="px-4 py-3 text-center">
          <Badge score={row.rag?.accuracy_score} />
        </td>
        <td className="px-4 py-3 text-center">
          <Badge score={row.graphrag?.accuracy_score} />
        </td>
        <td className="px-4 py-3 text-center">
          <Badge score={row.agentic?.accuracy_score} />
        </td>
        <td className="px-4 py-3 text-center text-xs text-gray-400 font-mono">
          {row.rag?.tokens_used || 0} / {row.graphrag?.tokens_used || 0} /{" "}
          {row.agentic?.tokens_used || 0}
        </td>
      </tr>
      {expanded && (
        <tr className="bg-gray-900/30">
          <td colSpan={7} className="px-6 py-4">
            <div className="grid grid-cols-3 gap-4 text-xs">
              {(["rag", "graphrag", "agentic"] as const).map((p) => (
                <div key={p} className="space-y-1">
                  <div className="font-semibold text-gray-300 uppercase">{p}</div>
                  <div className="text-gray-400">{row[p]?.answer || "—"}</div>
                  <div className="text-gray-600">
                    BERTScore: {(row[p]?.bertscore_f1 ?? 0).toFixed(3)} · {row[p]?.tokens_used ?? 0} tokens
                  </div>
                </div>
              ))}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ─── Token Metrics ────────────────────────────────────────────────────────────
function TokenMetrics({ data, stats }: { data: BenchmarkRecord[]; stats: SummaryStats | null }) {
  if (!data.length) return <EmptyState message="No benchmark data yet." />;

  const pipelines = [
    { key: "rag", label: "RAG", color: "#3b82f6" },
    { key: "graphrag", label: "GraphRAG", color: "#a855f7" },
    { key: "agentic", label: "Agentic", color: "#f97316" },
  ];

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

  const maxTokens = Math.max(...avgTokensByType.flatMap((d) => [d.rag, d.graphrag, d.agentic]));

  return (
    <div className="space-y-8">
      {/* Overall Stats Cards */}
      <div className="grid grid-cols-3 gap-4">
        {pipelines.map(({ key, label, color }) => {
          const s = stats?.[key as keyof SummaryStats] as any;
          return (
            <div key={key} className="bg-gray-900 rounded-xl border border-gray-800 p-5">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-3 h-3 rounded-full" style={{ background: color }} />
                <h3 className="font-semibold text-gray-200">{label}</h3>
              </div>
              <div className="space-y-3">
                <Stat label="Avg Tokens" value={`${Math.round(s?.avg_tokens ?? 0)}`} unit="tok" />
                <Stat label="Avg Latency" value={`${Math.round(s?.avg_latency_ms ?? 0)}`} unit="ms" />
                <Stat label="Accuracy" value={`${((s?.accuracy ?? 0) * 100).toFixed(1)}`} unit="%" />
              </div>
            </div>
          );
        })}
      </div>

      {/* Bar Chart by Question Type */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <h3 className="font-semibold text-gray-200 mb-6">Average Tokens by Question Type</h3>
        <div className="space-y-4">
          {avgTokensByType.map((row) => (
            <div key={row.qtype} className="space-y-1">
              <div className="text-xs text-gray-400 font-medium">{row.qtype}</div>
              <div className="space-y-1.5">
                {pipelines.map(({ key, label, color }) => {
                  const val = row[key as keyof typeof row] as number;
                  const pct = (val / maxTokens) * 100;
                  return (
                    <div key={key} className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 w-20 text-right">{label}</span>
                      <div className="flex-1 bg-gray-800 rounded-full h-5 relative overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-700 flex items-center justify-end pr-2"
                          style={{ width: `${pct}%`, background: color }}
                        >
                          <span className="text-xs text-white font-mono">{Math.round(val)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-xs text-gray-600">
          ⚡ Lower token usage = more efficient. Agentic costs more but answers harder questions.
        </div>
      </div>

      {/* Distribution histogram */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <h3 className="font-semibold text-gray-200 mb-4">Token Distribution</h3>
        <TokenHistogram data={data} />
      </div>
    </div>
  );
}

function TokenHistogram({ data }: { data: BenchmarkRecord[] }) {
  const buckets = [0, 200, 500, 1000, 2000, 5000, Infinity];
  const labels = ["0-200", "200-500", "500-1k", "1k-2k", "2k-5k", "5k+"];

  const pipelines = [
    { key: "rag", label: "RAG", color: "#3b82f6" },
    { key: "graphrag", label: "GraphRAG", color: "#a855f7" },
    { key: "agentic", label: "Agentic", color: "#f97316" },
  ];

  const bucketCounts = labels.map((_, i) =>
    Object.fromEntries(
      pipelines.map(({ key }) => [
        key,
        data.filter((r) => {
          const t = r[key as keyof BenchmarkRecord] as any;
          const tok = t?.tokens_used ?? 0;
          return tok >= buckets[i] && tok < buckets[i + 1];
        }).length,
      ])
    )
  );

  const maxCount = Math.max(...bucketCounts.flatMap((b) => Object.values(b) as number[]));

  return (
    <div className="flex items-end gap-2 h-32">
      {labels.map((label, i) => (
        <div key={label} className="flex-1 flex flex-col items-center gap-1">
          <div className="w-full flex items-end gap-0.5 h-24">
            {pipelines.map(({ key, color }) => {
              const count = (bucketCounts[i][key] as number) || 0;
              const height = maxCount ? (count / maxCount) * 100 : 0;
              return (
                <div
                  key={key}
                  className="flex-1 rounded-t transition-all"
                  style={{ height: `${height}%`, background: color, opacity: 0.8 }}
                  title={`${key}: ${count}`}
                />
              );
            })}
          </div>
          <span className="text-xs text-gray-500">{label}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Accuracy Panel ───────────────────────────────────────────────────────────
function AccuracyPanel({ data, stats }: { data: BenchmarkRecord[]; stats: SummaryStats | null }) {
  if (!data.length) return <EmptyState message="No benchmark data yet." />;
  const qtypes = [...new Set(data.map((r) => r.qtype))].sort();

  return (
    <div className="space-y-6">
      {/* Accuracy by qtype heatmap */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <h3 className="font-semibold text-gray-200 mb-6">Accuracy by Question Type</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 text-xs">
                <th className="text-left py-2 pr-6">Question Type</th>
                <th className="text-center py-2 px-4">RAG</th>
                <th className="text-center py-2 px-4">GraphRAG</th>
                <th className="text-center py-2 px-4">Agentic</th>
                <th className="text-center py-2 px-4">Δ (Agentic - RAG)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {qtypes.map((qt) => {
                const subset = data.filter((r) => r.qtype === qt);
                const ragAcc = accRate(subset, "rag");
                const gragAcc = accRate(subset, "graphrag");
                const agAcc = accRate(subset, "agentic");
                const delta = agAcc - ragAcc;
                return (
                  <tr key={qt} className="hover:bg-gray-800/30">
                    <td className="py-3 pr-6">
                      <span className="px-2 py-0.5 bg-gray-800 text-gray-300 rounded text-xs">{qt}</span>
                      <span className="ml-2 text-xs text-gray-600">({subset.length}q)</span>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <AccuracyCell value={ragAcc} />
                    </td>
                    <td className="py-3 px-4 text-center">
                      <AccuracyCell value={gragAcc} />
                    </td>
                    <td className="py-3 px-4 text-center">
                      <AccuracyCell value={agAcc} />
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`text-sm font-bold ${delta > 0 ? "text-green-400" : delta < 0 ? "text-red-400" : "text-gray-500"}`}>
                        {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* BERTScore comparison */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <h3 className="font-semibold text-gray-200 mb-4">BERTScore F1 Distribution</h3>
        <div className="grid grid-cols-3 gap-4">
          {(["rag", "graphrag", "agentic"] as const).map((p) => {
            const scores = data.map((r) => r[p]?.bertscore_f1 ?? 0).filter((v) => v > 0);
            const avgScore = scores.length ? avg(scores) : 0;
            return (
              <div key={p} className="text-center">
                <div className="text-3xl font-bold text-white">{avgScore.toFixed(3)}</div>
                <div className="text-sm text-gray-400 mt-1 capitalize">{p}</div>
                <div className="text-xs text-gray-600">avg BERTScore F1</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function AccuracyCell({ value }: { value: number }) {
  const bg = value >= 0.8 ? "bg-green-900 text-green-300" : value >= 0.5 ? "bg-yellow-900 text-yellow-300" : "bg-red-900 text-red-300";
  return (
    <span className={`px-3 py-1 rounded text-xs font-bold ${bg}`}>
      {(value * 100).toFixed(1)}%
    </span>
  );
}

// ─── Agent Trace Viewer ───────────────────────────────────────────────────────
function AgentTracePanel() {
  const [traces, setTraces] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);

  useEffect(() => {
    api.getLog("agent_invocations", 200).then(setTraces).catch(console.error);
  }, []);

  const grouped: Record<string, any[]> = {};
  for (const t of traces) {
    const qid = t.question_id || "unknown";
    if (!grouped[qid]) grouped[qid] = [];
    grouped[qid].push(t);
  }
  const qids = Object.keys(grouped);

  return (
    <div className="grid grid-cols-3 gap-6 h-[70vh]">
      {/* Question list */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-y-auto">
        <div className="px-4 py-3 border-b border-gray-800 text-sm font-semibold text-gray-300">
          Questions ({qids.length})
        </div>
        {qids.length === 0 && <EmptyState message="No agent traces yet." />}
        {qids.map((qid) => (
          <button
            key={qid}
            onClick={() => setSelected(grouped[qid])}
            className={`w-full text-left px-4 py-3 text-sm border-b border-gray-800 transition ${
              selected === grouped[qid] ? "bg-orange-900/30 text-orange-300" : "hover:bg-gray-800 text-gray-300"
            }`}
          >
            <div className="font-mono text-xs text-gray-500">{qid}</div>
            <div className="text-xs text-gray-400">{grouped[qid].length} agent calls</div>
          </button>
        ))}
      </div>

      {/* Trace detail */}
      <div className="col-span-2 bg-gray-900 rounded-xl border border-gray-800 overflow-y-auto">
        {!selected ? (
          <div className="h-full flex items-center justify-center text-gray-600">
            Select a question to view its investigation trace
          </div>
        ) : (
          <div className="p-4 space-y-2">
            {selected.map((step: any, i: number) => (
              <TraceStepCard key={i} step={step} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function TraceStepCard({ step, index }: { step: any; index: number }) {
  const [open, setOpen] = useState(false);
  const agentColors: Record<string, string> = {
    EntityLinkerAgent: "bg-blue-900 text-blue-300",
    GraphTraversalAgent: "bg-purple-900 text-purple-300",
    SimilaritySearchAgent: "bg-cyan-900 text-cyan-300",
    DocumentRetrievalAgent: "bg-green-900 text-green-300",
    AggregationAgent: "bg-yellow-900 text-yellow-300",
    EvidenceEvaluatorAgent: "bg-pink-900 text-pink-300",
    LedgerGuard: "bg-red-900 text-red-300",
  };

  const colorClass = agentColors[step.agent_name] || "bg-gray-800 text-gray-300";

  return (
    <div className="rounded-lg border border-gray-800 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-800/50 transition text-left"
      >
        <span className="text-xs text-gray-600 font-mono w-6">{index + 1}</span>
        <span className={`px-2 py-0.5 rounded text-xs font-mono ${colorClass}`}>
          {step.agent_name}
        </span>
        <span className="text-xs text-gray-400 flex-1 truncate">{step.input_summary}</span>
        <span className="text-xs text-gray-600 font-mono">{step.tokens_used}t · {step.latency_ms}ms</span>
        <span className="text-gray-600">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="px-4 pb-4 bg-gray-900/50 text-xs space-y-2">
          <div>
            <span className="text-gray-500">Output: </span>
            <span className="text-gray-300">{step.output_summary}</span>
          </div>
          <div className="text-gray-600">{step.timestamp}</div>
        </div>
      )}
    </div>
  );
}

// ─── Confusion Matrix ─────────────────────────────────────────────────────────
function ConfusionMatrixPanel() {
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-8">
      <h3 className="font-semibold text-gray-200 mb-2">Ladder Routing Accuracy — Confusion Matrix</h3>
      <p className="text-sm text-gray-500 mb-6">
        Ground truth = cheapest rung that answered correctly. Predicted = what the Ladder routed to.
      </p>
      <div className="overflow-x-auto">
        <table className="text-sm">
          <thead>
            <tr>
              <th className="px-4 py-2 text-right text-gray-500">True → / Predicted ↓</th>
              {["RAG", "GraphRAG", "Agentic", "none_passed"].map((col) => (
                <th key={col} className="px-6 py-2 text-center text-gray-400">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {["RAG", "GraphRAG", "Agentic"].map((row) => (
              <tr key={row} className="border-t border-gray-800">
                <td className="px-4 py-3 text-right text-gray-400">{row}</td>
                {["RAG", "GraphRAG", "Agentic", "none_passed"].map((col) => (
                  <td key={col} className="px-6 py-3 text-center">
                    <span className={`px-3 py-1 rounded text-sm font-bold ${row === col ? "bg-green-900 text-green-300" : "bg-gray-800 text-gray-500"}`}>
                      —
                    </span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-gray-600 mt-4">
        Run the full benchmark to populate this matrix. Values are computed from benchmark_results.jsonl.
      </p>
    </div>
  );
}

// ─── Query Playground ─────────────────────────────────────────────────────────
function QueryPlayground() {
  const [question, setQuestion] = useState("");
  const [pipeline, setPipeline] = useState("all");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const EXAMPLES = [
    "Who won the gold medal in the men's 20 kilometres walk at the 2012 Summer Olympics?",
    "How many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
    "Which sailing event at the 2000 Summer Olympics had the highest number of competitors?",
    "Who won the gold medal in the women's 57 kg judo event at the Summer Olympics immediately before 2020?",
  ];

  const handleSubmit = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const res = await api.query(question, pipeline);
      setResults(res.results);
    } catch (e: any) {
      setError(e.message);
    }
    setLoading(false);
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <h3 className="font-semibold text-gray-200 mb-4">Live Query Interface</h3>

        {/* Examples */}
        <div className="mb-4 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => setQuestion(ex)}
              className="px-3 py-1 text-xs bg-gray-800 text-gray-400 rounded-lg hover:bg-gray-700 hover:text-gray-200 transition truncate max-w-xs"
              title={ex}
            >
              {ex.slice(0, 50)}...
            </button>
          ))}
        </div>

        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask any question about the Olympic corpus..."
          rows={3}
          className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-orange-500 resize-none"
        />

        <div className="mt-3 flex gap-3 items-center">
          <select
            value={pipeline}
            onChange={(e) => setPipeline(e.target.value)}
            className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-300"
          >
            <option value="all">All 3 Pipelines</option>
            <option value="rag">RAG only</option>
            <option value="graphrag">GraphRAG only</option>
            <option value="agentic">Agentic only</option>
          </select>
          <button
            onClick={handleSubmit}
            disabled={loading || !question.trim()}
            className="px-6 py-2 bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-white text-sm rounded-lg transition font-medium"
          >
            {loading ? "Investigating..." : "Run Query →"}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {results && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(results).map(([pipeline, result]: [string, any]) => (
            <PipelineResultCard key={pipeline} pipeline={pipeline} result={result} />
          ))}
        </div>
      )}
    </div>
  );
}

function PipelineResultCard({ pipeline, result }: { pipeline: string; result: any }) {
  const colors: Record<string, string> = {
    rag: "border-blue-700",
    graphrag: "border-purple-700",
    agentic: "border-orange-700",
  };
  const labels: Record<string, string> = { rag: "RAG", graphrag: "GraphRAG", agentic: "Agentic" };

  return (
    <div className={`bg-gray-900 rounded-xl border-2 ${colors[pipeline] || "border-gray-700"} p-5`}>
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-gray-200">{labels[pipeline] || pipeline}</h4>
        {result.exit_type && (
          <span className={`text-xs px-2 py-0.5 rounded font-mono ${
            result.exit_type === "ANSWER" ? "bg-green-900 text-green-300" :
            result.exit_type === "PARTIAL" ? "bg-yellow-900 text-yellow-300" :
            "bg-red-900 text-red-300"
          }`}>
            {result.exit_type}
          </span>
        )}
      </div>
      <p className="text-sm text-gray-300 leading-relaxed mb-4">{result.answer}</p>
      <div className="flex gap-4 text-xs text-gray-500 border-t border-gray-800 pt-3">
        <span>🪙 {result.tokens_used} tokens</span>
        <span>⏱ {result.latency_ms}ms</span>
        {result.n_sources_cited != null && <span>📎 {result.n_sources_cited} sources</span>}
      </div>
    </div>
  );
}

// ─── Shared Components ────────────────────────────────────────────────────────
function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center h-48 text-gray-500">
      <div className="animate-spin w-8 h-8 border-2 border-orange-500 border-t-transparent rounded-full" />
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm text-center px-4">
      <span className="text-4xl mb-3">📭</span>
      {message}
    </div>
  );
}

function Stat({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="flex justify-between items-baseline">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="font-mono text-sm text-white">
        {value} <span className="text-gray-500 text-xs">{unit}</span>
      </span>
    </div>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function avg(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function accRate(data: BenchmarkRecord[], pipeline: string): number {
  const subset = data.filter((r) => (r as any)[pipeline]?.accuracy_score);
  if (!subset.length) return 0;
  return subset.filter((r) => (r as any)[pipeline].accuracy_score === "PASS").length / subset.length;
}
