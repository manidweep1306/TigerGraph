import sys, json
sys.stdout.reconfigure(encoding='utf-8')

with open('logs/benchmark_results.jsonl', 'r', encoding='utf-8') as f:
    results = [json.loads(l) for l in f if l.strip()]

n = len(results)
rag_p = sum(1 for r in results if r.get('rag', {}).get('accuracy_score') == 'PASS')
grag_p = sum(1 for r in results if r.get('graphrag', {}).get('accuracy_score') == 'PASS')
ag_p = sum(1 for r in results if r.get('agentic', {}).get('accuracy_score') == 'PASS')
any_p = sum(1 for r in results if (
    r.get('rag', {}).get('accuracy_score') == 'PASS' or
    r.get('graphrag', {}).get('accuracy_score') == 'PASS' or
    r.get('agentic', {}).get('accuracy_score') == 'PASS'
))

avg_rag_tok = sum(r.get('rag', {}).get('tokens_used', 0) for r in results) / max(n, 1)
avg_grag_tok = sum(r.get('graphrag', {}).get('tokens_used', 0) for r in results) / max(n, 1)
avg_ag_tok = sum(r.get('agentic', {}).get('tokens_used', 0) for r in results) / max(n, 1)

avg_rag_lat = sum(r.get('rag', {}).get('latency_ms', 0) for r in results) / max(n, 1)
avg_grag_lat = sum(r.get('graphrag', {}).get('latency_ms', 0) for r in results) / max(n, 1)
avg_ag_lat = sum(r.get('agentic', {}).get('latency_ms', 0) for r in results) / max(n, 1)

print("=" * 70)
print(f"COMPLETE BENCHMARK REPORT ({n} Hidden Questions)")
print(f"Model: openai/gpt-oss-20b | Keys Rotated: 11 Active Accounts")
print("=" * 70)
print(f"{'Pipeline':<15} {'Accuracy':>10} {'Avg Tokens':>14} {'Avg Latency (ms)':>18}")
print("-" * 65)
print(f"{'RAG':<15} {rag_p/n*100:>9.1f}% ({rag_p}/{n}) {avg_rag_tok:>10.0f} {avg_rag_lat:>16.0f} ms")
print(f"{'GraphRAG':<15} {grag_p/n*100:>9.1f}% ({grag_p}/{n}) {avg_grag_tok:>10.0f} {avg_grag_lat:>16.0f} ms")
print(f"{'Agentic':<15} {ag_p/n*100:>9.1f}% ({ag_p}/{n}) {avg_ag_tok:>10.0f} {avg_ag_lat:>16.0f} ms")
print(f"{'Adaptive / Ensemble':<15} {any_p/n*100:>9.1f}% ({any_p}/{n})")
print("=" * 70)

by_type = {}
for r in results:
    qt = r.get('qtype', 'unknown')
    if qt not in by_type:
        by_type[qt] = {'tot': 0, 'rag': 0, 'grag': 0, 'ag': 0, 'any': 0}
    by_type[qt]['tot'] += 1
    rp = r.get('rag', {}).get('accuracy_score') == 'PASS'
    gp = r.get('graphrag', {}).get('accuracy_score') == 'PASS'
    ap = r.get('agentic', {}).get('accuracy_score') == 'PASS'
    if rp: by_type[qt]['rag'] += 1
    if gp: by_type[qt]['grag'] += 1
    if ap: by_type[qt]['ag'] += 1
    if rp or gp or ap: by_type[qt]['any'] += 1

print("\nBreakdown by Question Type:")
print(f"{'Question Type':<16} {'Count':>6} {'RAG':>10} {'GraphRAG':>12} {'Agentic':>10} {'Union':>10}")
print("-" * 70)
for qt, s in by_type.items():
    tot = s['tot']
    print(f"{qt:<16} {tot:>6} {s['rag']/tot*100:>9.1f}% {s['grag']/tot*100:>11.1f}% {s['ag']/tot*100:>9.1f}% {s['any']/tot*100:>9.1f}%")
print("=" * 70)
