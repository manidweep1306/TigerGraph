import React from "react";
import { Icons } from "./Icons";

export function LoadingSpinner({ label = "Loading data..." }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 space-y-3 text-slate-500">
      <div className="animate-spin w-8 h-8 border-2 border-[#FF7A00] border-t-transparent rounded-full shadow-xs" />
      <span className="text-xs font-mono font-semibold">{label}</span>
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-56 bg-white rounded-xl border border-slate-200 p-6 text-center space-y-2 shadow-xs">
      <div className="text-orange-500"><Icons.Benchmark /></div>
      <p className="text-xs sm:text-sm text-slate-600 max-w-sm font-medium">{message}</p>
    </div>
  );
}

export function MetricRow({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="flex justify-between items-baseline text-xs">
      <span className="text-slate-600 font-medium">{label}</span>
      <span className="font-mono text-xs font-bold text-slate-900">
        {value} <span className="text-[10px] text-slate-400 font-normal">{unit}</span>
      </span>
    </div>
  );
}

export function AccuracyPill({ value }: { value: number }) {
  const pct = (value * 100).toFixed(1);
  if (value >= 0.8) {
    return (
      <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-emerald-50 text-emerald-800 border border-emerald-200 shadow-2xs">
        {pct}%
      </span>
    );
  }
  if (value >= 0.5) {
    return (
      <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-amber-50 text-amber-800 border border-amber-200 shadow-2xs">
        {pct}%
      </span>
    );
  }
  return (
    <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-rose-50 text-rose-800 border border-rose-200 shadow-2xs">
      {pct}%
    </span>
  );
}

export function avg(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}
