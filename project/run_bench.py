import os
import sys
from pathlib import Path

sys.path.insert(0, r"d:\TigerGraph\project")
from backend.evaluation.benchmark_runner import run_benchmark

print("Starting benchmark (auto-resuming remaining questions)...")
run_benchmark(r"d:\TigerGraph\project\data\questions\eval_hidden.jsonl")
print("Done!")
