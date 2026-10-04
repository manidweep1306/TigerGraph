import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from backend.pipelines import rag_pipeline, graphrag_pipeline
from backend.evaluation.evaluator import llm_judge

gold_map = {}
for line in open('data/questions/eval_hidden_gold.jsonl', encoding='utf-8'):
    d = json.loads(line)
    gold_map[d['qid']] = d

test_qids = ['eval-001', 'eval-002', 'eval-003', 'eval-005', 'eval-006', 'eval-007']
for qid in test_qids:
    qdata = gold_map[qid]
    q = qdata['question']
    g = qdata['answer']
    qt = qdata['qtype']
    print(f"=== {qid} ({qt}) ===")
    print("Q:", q)
    print("Gold:", g[:3])
    
    rag_res = rag_pipeline.run(q, qid)
    grag_res = graphrag_pipeline.run(q, qid)
    
    r_acc = llm_judge(q, g, rag_res['answer'])
    g_acc = llm_judge(q, g, grag_res['answer'])
    
    print(f"RAG  [{r_acc}]: {rag_res['answer'][:90]}")
    print(f"GRAG [{g_acc}]: {grag_res['answer'][:90]}")
    print("-" * 50)
