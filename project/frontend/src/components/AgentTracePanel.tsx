import React, { useState, useEffect, useMemo } from "react";
import { api } from "@/lib/api";
import { AgentTraceStep } from "@/types";
import { AGENT_BADGE_MAP } from "@/lib/constants";
import { Icons } from "./Icons";
import { LoadingSpinner } from "./Common";

export function AgentTracePanel() {
  const [traces, setTraces] = useState<AgentTraceStep[]>([]);
  const [selectedQid, setSelectedQid] = useState<string | null>(null);
  const [traceSearch, setTraceSearch] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .getLog("agent_invocations", 300)
      .then((data) => {
        const traceList: AgentTraceStep[] = Array.isArray(data) ? data : [];
        setTraces(traceList);
        if (traceList.length > 0) {
          setSelectedQid(traceList[0].question_id || "unknown");
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const grouped = useMemo(() => {
    const map: Record<string, AgentTraceStep[]> = {};
    for (const t of traces) {
      const qid = t.question_id || "unknown";
      if (!map[qid]) map[qid] = [];
      map[qid].push(t);
    }
    return map;
  }, [traces]);

  const qids = useMemo(() => {
    return Object.keys(grouped).filter((qid) =>
      traceSearch ? qid.toLowerCase().includes(traceSearch.toLowerCase()) : true
    );
  }, [grouped, traceSearch]);

  const currentTrace = selectedQid ? grouped[selectedQid] || [] : [];

  if (loading) return <LoadingSpinner label="Fetching agent investigation telemetry traces..." />;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[72vh]">
      {/* Question Selector Sidebar */}
      <div className="bg-white rounded-xl border border-slate-200 flex flex-col overflow-hidden shadow-sm">
        <div className="p-3.5 border-b border-slate-200 space-y-2 bg-slate-50">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-xs text-slate-800">Investigation Sessions</h3>
            <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-slate-200 text-slate-700 border border-slate-300">
              {qids.length} Sessions
            </span>
          </div>
          <input
            value={traceSearch}
            onChange={(e) => setTraceSearch(e.target.value)}
            placeholder="Search trace QID..."
            className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-[#FF7A00]"
          />
        </div>

        <div className="overflow-y-auto flex-1 divide-y divide-slate-100">
          {qids.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-500">No traces match query</div>
          ) : (
            qids.map((qid) => {
              const isSelected = selectedQid === qid;
              const steps = grouped[qid];
              const totalTok = steps.reduce((sum, s) => sum + (s.tokens_used || 0), 0);
              const totalLat = steps.reduce((sum, s) => sum + (s.latency_ms || 0), 0);
              return (
                <button
                  key={qid}
                  onClick={() => setSelectedQid(qid)}
                  className={`w-full text-left p-3.5 transition flex flex-col gap-1 outline-none cursor-pointer ${
                    isSelected
                      ? "bg-[#FFF5EB] border-l-4 border-[#FF7A00] text-[#E65100] font-semibold"
                      : "hover:bg-slate-50 text-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-slate-900">{qid}</span>
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 font-mono border border-slate-200">
                      {steps.length} steps
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono">
                    <span>🪙 {totalTok} tok</span>
                    <span>•</span>
                    <span>⏱ {totalLat}ms</span>
                  </div>
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Trace Timeline & Step Detail */}
      <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 flex flex-col overflow-hidden shadow-sm">
        <div className="p-3.5 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-500">Viewing Trace:</span>
            <span className="text-xs font-mono font-bold text-orange-600">
              {selectedQid || "None Selected"}
            </span>
          </div>
          {currentTrace.length > 0 && (
            <span className="text-[11px] text-slate-500 font-mono font-semibold">
              Investigation Chain ({currentTrace.length} calls)
            </span>
          )}
        </div>

        <div className="overflow-y-auto flex-1 p-4 space-y-3 bg-slate-50/50">
          {currentTrace.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 text-xs text-center p-8 space-y-2">
              <span className="text-3xl text-orange-400"><Icons.Trace /></span>
              <p className="text-slate-600 font-medium">
                Select a question session from the sidebar to inspect its agent step-by-step telemetry.
              </p>
            </div>
          ) : (
            currentTrace.map((step, idx) => (
              <TraceStepCard key={idx} step={step} index={idx} total={currentTrace.length} />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function TraceStepCard({ step, index, total }: { step: AgentTraceStep; index: number; total: number }) {
  const [open, setOpen] = useState(false);
  const badgeInfo = AGENT_BADGE_MAP[step.agent_name] || {
    label: step.agent_name || "SpecializedAgent",
    bg: "bg-slate-100",
    text: "text-slate-800",
    border: "border-slate-200",
  };

  const isLast = index === total - 1;

  return (
    <div className="relative">
      {/* Visual Connected Data Stream Connector */}
      {!isLast && (
        <div className="absolute left-[26px] top-[40px] bottom-[-16px] w-[2px] bg-slate-200 z-0">
          <div className="w-full h-full spell-beam opacity-40"></div>
        </div>
      )}

      <div className={`relative z-10 rounded-xl border bg-white overflow-hidden shadow-2xs transition-all duration-200 hover:shadow-md ${open ? "border-orange-300 ring-1 ring-orange-200" : "border-slate-200 hover:border-slate-300"}`}>
        <button
          onClick={() => setOpen(!open)}
          className="w-full flex items-center justify-between p-3.5 text-left hover:bg-slate-50/80 transition-colors outline-none gap-3 cursor-pointer group"
        >
          <div className="flex items-center gap-3 min-w-0">
            <div className={`w-7 h-7 rounded-full border flex items-center justify-center text-[10px] font-mono font-bold shrink-0 transition-all duration-200 ${open ? "bg-[#FF7A00] text-white border-[#FF7A00] scale-110 shadow-xs" : "bg-slate-100 text-slate-700 border-slate-300 group-hover:border-orange-400"}`}>
              {index + 1}
            </div>
            <span
              className={`px-2.5 py-0.5 rounded-md text-[11px] font-mono font-bold border ${badgeInfo.bg} ${badgeInfo.text} ${badgeInfo.border} shrink-0 shadow-2xs transition-transform duration-200 group-hover:scale-105`}
            >
              {badgeInfo.label}
            </span>
            <span className="text-xs text-slate-800 font-medium truncate max-w-sm group-hover:text-slate-900">
              {step.input_summary || step.action || "Executed task"}
            </span>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <span className="text-[11px] font-mono font-semibold text-slate-600 bg-slate-100 px-2 py-0.5 rounded border border-slate-200 transition-colors group-hover:bg-slate-200">
              {step.tokens_used ?? 0}t · {step.latency_ms ?? 0}ms
            </span>
            <span className="text-slate-400 group-hover:text-slate-600 transition-transform duration-200">
              {open ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
            </span>
          </div>
        </button>

        {open && (
          <div className="px-4 pb-4 pt-2 border-t border-slate-100 bg-slate-50 text-xs space-y-2.5 font-sans">
            <div>
              <span className="text-slate-600 font-bold block text-[11px] uppercase tracking-wider mb-1">
                Task Output / Artifacts
              </span>
              <div className="bg-white p-3 rounded-lg border border-slate-200 text-slate-800 font-mono text-[11px] whitespace-pre-wrap shadow-inner">
                {step.output_summary || JSON.stringify(step.output || step.result || "No detailed payload", null, 2)}
              </div>
            </div>
            <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1">
              <span>Timestamp: {step.timestamp || "N/A"}</span>
              {step.strategy_changed && (
                <span className="text-amber-700 font-bold">⚠️ Dynamic Slot Revision Triggered</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
