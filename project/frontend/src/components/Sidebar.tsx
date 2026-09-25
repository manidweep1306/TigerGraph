import React from "react";
import { TabId } from "@/types";
import { Icons } from "./Icons";

interface NavItem {
  id: TabId;
  label: string;
  icon: React.ReactNode;
  count?: number;
}

interface SidebarProps {
  activeTab: TabId;
  onSelectTab: (id: TabId) => void;
  benchmarkCount: number;
}

export function Sidebar({ activeTab, onSelectTab, benchmarkCount }: SidebarProps) {
  const navItems: NavItem[] = [
    { id: "benchmark", label: "Benchmark Explorer", icon: <Icons.Benchmark />, count: benchmarkCount },
    { id: "tokens", label: "Token Efficiency", icon: <Icons.Token /> },
    { id: "accuracy", label: "Accuracy & BERTScore", icon: <Icons.Accuracy /> },
    { id: "trace", label: "Investigation Trace", icon: <Icons.Trace /> },
    { id: "matrix", label: "Routing Matrix", icon: <Icons.Matrix /> },
    { id: "playground", label: "Live Query Bench", icon: <Icons.Playground /> },
  ];

  return (
    <aside className="w-full md:w-72 bg-white border-r border-slate-200 flex flex-col shrink-0 min-h-screen shadow-sm z-20">
      {/* Brand Header */}
      <div className="px-5 py-[26px] min-h-[116px] border-b border-slate-200 flex items-center gap-4 bg-slate-50/40 group cursor-default select-none">
        <div className="relative w-14 h-14 rounded-2xl bg-gradient-to-br from-amber-400 via-orange-500 to-amber-600 p-0.5 shadow-lg shadow-orange-500/25 shrink-0 transition-all duration-300 group-hover:scale-105 group-hover:rotate-1 group-hover:shadow-orange-500/40">
          <div className="w-full h-full bg-slate-950 rounded-[14px] flex items-center justify-center text-3xl transition-transform duration-300 group-hover:scale-110">
            🐯
          </div>
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-orange-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-amber-500 border border-white"></span>
          </span>
        </div>
        <div>
          <h1 className="text-lg font-black tracking-tight bg-gradient-to-r from-orange-600 via-amber-600 to-yellow-600 bg-clip-text text-transparent leading-tight transition-all duration-300 group-hover:brightness-110">
            Agentic GraphRAG Evaluation Engine
          </h1>
        </div>
      </div>

      {/* Navigation Category */}
      <div className="flex-1 py-4 px-3 space-y-4 overflow-y-auto">
        <div>
          <div className="px-3 mb-2 text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
            <span>Investigation & Evaluation</span>
            <span className="w-1.5 h-1.5 rounded-full bg-orange-400/60 animate-pulse"></span>
          </div>
          <nav className="space-y-1" role="tablist" aria-orientation="vertical">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 text-xs font-semibold rounded-lg transition-all duration-200 text-left cursor-pointer group ${
                    isActive
                      ? "bg-[#FFF5EB] text-[#E65100] font-bold border-l-4 border-[#FF7A00] shadow-xs translate-x-1"
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/80 hover:translate-x-0.5"
                  }`}
                  role="tab"
                  aria-selected={isActive}
                >
                  <div className="flex items-center gap-2.5">
                    <span className={`transition-transform duration-200 ${isActive ? "text-[#FF7A00] scale-110" : "text-slate-400 group-hover:text-slate-600 group-hover:scale-105"}`}>
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </div>
                  {item.count !== undefined && item.count > 0 && (
                    <span
                      className={`text-[10px] font-mono px-1.5 py-0.2 rounded-full transition-colors ${
                        isActive ? "bg-orange-200 text-orange-900 font-bold" : "bg-slate-100 text-slate-500 group-hover:bg-slate-200"
                      }`}
                    >
                      {item.count}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="px-3.5 py-[14px] min-h-[48px] border-t border-slate-200 bg-slate-50/70 flex items-center justify-between">
        <span className="flex items-center gap-2 font-semibold text-slate-700 text-[13px]">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500 shadow-xs"></span>
          </span>
          Savanna Instance
        </span>
        <span className="text-[12px] font-mono text-emerald-800 bg-emerald-100 px-2 py-0.5 rounded-md font-extrabold border border-emerald-200 shadow-2xs transition-all hover:bg-emerald-200">
          CONNECTED
        </span>
      </div>
    </aside>
  );
}
