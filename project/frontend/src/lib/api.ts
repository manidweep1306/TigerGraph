/**
 * API client for the Agentic GraphRAG backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

import { BenchmarkRecord, PipelineResult, SummaryStats } from "@/types";

export type { BenchmarkRecord, PipelineResult, SummaryStats };

export interface TraceStep {
  step: string;
  step_id?: number;
  agent_type?: string;
  slot_id?: string;
  slot_state_before?: string;
  slot_state_after?: string;
  actual_gain?: number;
  tokens_used?: number;
  confidence?: number;
  entailment?: string;
}

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json();
}

export const api = {
  health: () => fetchAPI<{ status: string; baseline_fairness: string }>("/health"),

  query: (question: string, pipeline = "all", questionId?: string) =>
    fetchAPI<{ question: string; question_id: string; results: Record<string, PipelineResult> }>(
      "/query",
      {
        method: "POST",
        body: JSON.stringify({ question, pipeline, question_id: questionId }),
      }
    ),

  adaptive: (question: string) =>
    fetchAPI<{ question: string; rung_path: string[]; answer: string; tokens_used: number }>(
      "/adaptive",
      { method: "POST", body: JSON.stringify({ question }) }
    ),

  getBenchmarkResults: (limit?: number) =>
    fetchAPI<BenchmarkRecord[]>(`/benchmark/results${limit ? `?limit=${limit}` : ""}`),

  getAdaptiveResults: (limit?: number) =>
    fetchAPI<any[]>(`/adaptive/results${limit ? `?limit=${limit}` : ""}`),

  getSummaryStats: () => fetchAPI<SummaryStats>("/stats/summary"),

  runBenchmark: (limit?: number) =>
    fetchAPI<{ status: string }>("/benchmark/run", {
      method: "POST",
      body: JSON.stringify({ limit }),
    }),

  getLog: (name: string, limit = 100) =>
    fetchAPI<any[]>(`/logs/${name}?limit=${limit}`),
};
