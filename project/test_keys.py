import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
keys = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]

for i, k in enumerate(keys):
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=k)
    try:
        r = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "1+1"}],
            max_tokens=3
        )
        print(f"Key {i+1} (...{k[-4:]}): ACTIVE (answer: {r.choices[0].message.content.strip()})")
    except Exception as e:
        err = str(e)
        if "rate_limit" in err or "429" in err:
            print(f"Key {i+1} (...{k[-4:]}): 429 Rate limited ({err[:80]}...)")
        else:
            print(f"Key {i+1} (...{k[-4:]}): Error: {err[:80]}")
