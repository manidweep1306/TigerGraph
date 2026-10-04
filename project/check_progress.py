import sys, json
sys.stdout.reconfigure(encoding='utf-8')
try:
    with open('logs/benchmark_results.jsonl', 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    n = len(lines)
    print(f"Completed: {n}/50")
    if n > 0:
        rag_p = sum(1 for r in lines if r.get('rag', {}).get('accuracy_score') == 'PASS')
        grag_p = sum(1 for r in lines if r.get('graphrag', {}).get('accuracy_score') == 'PASS')
        ag_p = sum(1 for r in lines if r.get('agentic', {}).get('accuracy_score') == 'PASS')
        print(f"RAG Accuracy: {rag_p}/{n} ({rag_p/n*100:.1f}%)")
        print(f"GraphRAG Accuracy: {grag_p}/{n} ({grag_p/n*100:.1f}%)")
        print(f"Agentic Accuracy: {ag_p}/{n} ({ag_p/n*100:.1f}%)")
        for r in lines[-3:]:
            qid = r['question_id']
            ra = r['rag']['accuracy_score']
            ga = r['graphrag']['accuracy_score']
            aa = r['agentic']['accuracy_score']
            print(f"QID {qid}: RAG={ra} GRAG={ga} AG={aa}")
except Exception as e:
    print("Status:", e)
