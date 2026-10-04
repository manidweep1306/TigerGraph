import json

lines = [json.loads(l) for l in open('data/questions/eval_hidden_gold.jsonl', encoding='utf-8') if l.strip()]
for d in lines:
    print(f"{d['qid']} ({d['qtype']}): {d['answer']}")
