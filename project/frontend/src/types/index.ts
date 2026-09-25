export type TabId = "benchmark" | "tokens" | "accuracy" | "trace" | "matrix" | "playground";

export interface PipelineResult {
  answer: string;
  sources?: string[];
  tokens_used: number;
  latency_ms: number;
  accuracy_score?: "PASS" | "FAIL";
  bertscore_f1?: number;
  exit_type?: "ANSWER" | "PARTIAL" | "ABSTAIN";
  n_chunks_retrieved?: number;
  n_sources_cited?: number;
  strategy_changed?: boolean;
  stop_reason?: string;
  trace?: AgentTraceStep[];
}

export interface BenchmarkRecord {
  question_id: string;
  question: string;
  qtype: string;
  rag?: PipelineResult;
  graphrag?: PipelineResult;
  agentic?: PipelineResult;
}

export interface PipelineSummary {
  accuracy: number;
  avg_tokens: number;
  avg_latency_ms: number;
  avg_bertscore: number;
  count: number;
}

export interface SummaryStats {
  total_questions: number;
  rag: PipelineSummary;
  graphrag: PipelineSummary;
  agentic: PipelineSummary;
  by_qtype: Record<string, Record<string, number>>;
}

export interface HealthResponse {
  status: string;
  baseline_fairness?: string;
  model?: string;
  error?: string;
}

export interface AgentTraceStep {
  agent_name: string;
  action?: string;
  input_summary?: string;
  output_summary?: string;
  tokens_used?: number;
  latency_ms?: number;
  timestamp?: string;
  strategy_changed?: boolean;
  output?: unknown;
  result?: unknown;
  question_id?: string;
}

export interface ConfusionMatrixData {
  matrix: Record<string, Record<string, number>>;
  total: number;
  accuracy: number;
}

export interface PipelineConfigItem {
  id: string;
  name: string;
  tag: string;
  color: string;
  pillBg: string;
  badgeClass: string;
  accentBorder: string;
}
