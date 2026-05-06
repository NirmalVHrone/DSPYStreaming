"""
Quick smoke-test for both endpoints.
Run the server first:  uvicorn main:app --reload
Then run:              python test_client.py
"""

import asyncio
import httpx

BASE = "http://127.0.0.1:8000"
QUESTION = "What is the capital of France and why is it historically significant?"


async def test_normal():
    print("=" * 60)
    print("NORMAL endpoint  POST /predict")
    print("=" * 60)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{BASE}/predict", json={"question": QUESTION})
        resp.raise_for_status()
        data = resp.json()
    print(f"Question : {data['question']}")
    print(f"Answer   : {data['answer']}")
    print()


async def test_streaming():
    print("=" * 60)
    print("STREAMING endpoint  POST /predict/stream  (SSE)")
    print("=" * 60)
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            f"{BASE}/predict/stream",
            json={"question": QUESTION},
        ) as resp:
            resp.raise_for_status()
            print("Tokens  : ", end="", flush=True)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    print()
                    print("Stream finished.")
                    break
                import json
                data = json.loads(payload)
                if "delta" in data:
                    print(data["delta"], end="", flush=True)
                elif "answer" in data:
                    print(f"\nFinal   : {data['answer']}")
    print()


async def main():
    await test_normal()
    await test_streaming()


if __name__ == "__main__":
    asyncio.run(main())
