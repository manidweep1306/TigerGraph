export const PIPELINE_CONFIG = {
  rag: {
    id: "rag",
    name: "RAG",
    tag: "Vector Search",
    color: "#3b82f6", // blue-500
    badgeClass: "bg-blue-50 text-blue-700 border-blue-200",
    accentBorder: "border-blue-200",
  },
  graphrag: {
    id: "graphrag",
    name: "GraphRAG",
    tag: "Graph + Vector",
    color: "#8b5cf6", // purple-500
    badgeClass: "bg-purple-50 text-purple-700 border-purple-200",
    accentBorder: "border-purple-200",
  },
  agentic: {
    id: "agentic",
    name: "Agentic GraphRAG",
    tag: "VoI Orchestrated",
    color: "#10b981", // emerald-500
    badgeClass: "bg-emerald-50 text-emerald-700 border-emerald-200",
    accentBorder: "border-emerald-200",
  },
} as const;

export const AGENT_BADGE_MAP: Record<
  string,
  { label: string; bg: string; text: string; border: string }
> = {
  entity_linker: {
    label: "Entity Linker",
    bg: "bg-indigo-50",
    text: "text-indigo-700",
    border: "border-indigo-200",
  },
  graph_traversal: {
    label: "Graph Traversal",
    bg: "bg-purple-50",
    text: "text-purple-700",
    border: "border-purple-200",
  },
  similarity_search: {
    label: "Similarity Search",
    bg: "bg-blue-50",
    text: "text-blue-700",
    border: "border-blue-200",
  },
  document_retrieval: {
    label: "Document Retrieval",
    bg: "bg-sky-50",
    text: "text-sky-700",
    border: "border-sky-200",
  },
  aggregation: {
    label: "Aggregation Agent",
    bg: "bg-amber-50",
    text: "text-amber-700",
    border: "border-amber-200",
  },
  evidence_evaluator: {
    label: "Evidence Evaluator",
    bg: "bg-emerald-50",
    text: "text-emerald-700",
    border: "border-emerald-200",
  },
  decomposer: {
    label: "Decomposer",
    bg: "bg-rose-50",
    text: "text-rose-700",
    border: "border-rose-200",
  },
  synthesis: {
    label: "Synthesis Engine",
    bg: "bg-teal-50",
    text: "text-teal-700",
    border: "border-teal-200",
  },
};
