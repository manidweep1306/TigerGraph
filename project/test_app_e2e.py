import sys, urllib.request, json, time
sys.stdout.reconfigure(encoding="utf-8")

base_url = "http://127.0.0.1:8000"

print("1. Testing /health...")
h_res = urllib.request.urlopen(f"{base_url}/health")
h_data = json.loads(h_res.read().decode())
print("   Health Response:", h_data)
assert h_data["status"] == "healthy"
assert h_data["baseline_fairness"] == "PASS"
print("   [OK] /health: PASSED (Fairness: PASS)")

print("\n2. Testing /benchmark/results...")
b_res = urllib.request.urlopen(f"{base_url}/benchmark/results")
b_data = json.loads(b_res.read().decode())
print(f"   Retrieved {len(b_data)} benchmark records.")
assert len(b_data) == 50
print("   First record sample:", {k: b_data[0][k] for k in ("question_id", "question", "qtype")})
print("   [OK] /benchmark/results: PASSED (50 records ready for UI table)")

print("\n3. Testing /stats/summary...")
s_res = urllib.request.urlopen(f"{base_url}/stats/summary")
s_data = json.loads(s_res.read().decode())
print("   Summary Stats keys:", list(s_data.keys()))
print("   RAG Stats:", s_data.get("rag"))
print("   GraphRAG Stats:", s_data.get("graphrag"))
print("   Agentic Stats:", s_data.get("agentic"))
print("   [OK] /stats/summary: PASSED (Ready for UI charts)")

print("\n4. Testing Live /query on multi-hop question...")
q_req = urllib.request.Request(
    f"{base_url}/query",
    data=json.dumps({
        "question": "Who won the gold medal in the event held at San Sicario on February 15, 2006?",
        "pipeline": "all"
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
t0 = time.time()
q_res = urllib.request.urlopen(q_req)
q_data = json.loads(q_res.read().decode())
dt = time.time() - t0
print(f"   Live Query completed in {dt:.2f}s")
print("   RAG Answer     :", q_data["results"]["rag"]["answer"][:80])
print("   GraphRAG Answer:", q_data["results"]["graphrag"]["answer"][:80])
print("   Agentic Answer :", q_data["results"]["agentic"]["answer"][:80])
print("   [OK] /query: PASSED (All 3 pipelines live)")

print("\n5. Testing /adaptive ladder query...")
ad_req = urllib.request.Request(
    f"{base_url}/adaptive",
    data=json.dumps({
        "question": "How many nations competed in Fencing at the 1988 Summer Olympics – Men's foil?"
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
ad_res = urllib.request.urlopen(ad_req)
ad_data = json.loads(ad_res.read().decode())
print("   Adaptive Ladder Path:", ad_data.get("rung_path"))
print("   Adaptive Answer     :", ad_data.get("answer"))
print("   [OK] /adaptive: PASSED")

print("\n" + "=" * 60)
print("ALL APPLICATION SERVICES ARE FULLY OPERATIONAL!")
print("=" * 60)
