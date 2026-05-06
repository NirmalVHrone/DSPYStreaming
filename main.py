import json
import os
from typing import AsyncGenerator

import dspy
import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# DSPy setup  (used by the normal /predict endpoint)
# ---------------------------------------------------------------------------

_MODEL = os.getenv("LM_MODEL", "openai/gpt-4o-mini")

lm = dspy.LM(
    model=_MODEL,
    api_key=os.getenv("OPENAI_API_KEY"),
    cache=False,
)
dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# DSPy signatures & modules
# ---------------------------------------------------------------------------

class QASignature(dspy.Signature):
    """Answer the question thoroughly and clearly."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class QAModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(QASignature)

    def forward(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)


qa_module = QAModule()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DSPy FastAPI Demo",
    description="Normal and streaming response endpoints powered by DSPy",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class NormalResponse(BaseModel):
    question: str
    answer: str


# ---------------------------------------------------------------------------
# Endpoint 1: Normal (non-streaming) response
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=NormalResponse, summary="Normal Q&A response")
async def predict(body: QuestionRequest) -> NormalResponse:
    """
    Returns the complete answer in a single JSON response after DSPy
    finishes the full generation.
    """
    try:
        result = qa_module(question=body.question)
        return NormalResponse(question=body.question, answer=result.answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoint 2: Streaming response
# ---------------------------------------------------------------------------

async def _token_generator(question: str) -> AsyncGenerator[str, None]:
    """
    Calls litellm.acompletion with stream=True and yields one SSE event per
    token.  litellm is DSPy's own transport layer, so no extra dependency is
    added.  We bypass dspy.streamify here because that layer has to buffer
    tokens while it scans for its [[ ## field ## ]] markers — which defeats
    real-time streaming.

    SSE events emitted:
      data: {"delta": "<token>"}   — one per LLM token
      data: {"answer": "<full>"}   — once, after the last token
      data: [DONE]                 — signals end of stream
    """
    messages = [
        {
            "role": "system",
            "content": "Answer the question thoroughly and clearly.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = await litellm.acompletion(
        model=_MODEL,
        messages=messages,
        stream=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    collected: list[str] = []

    async for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            collected.append(token)
            yield f"data: {json.dumps({'delta': token})}\n\n"

    # Final assembled answer
    yield f"data: {json.dumps({'answer': ''.join(collected)})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/predict/stream", summary="Streaming Q&A response (SSE)")
async def predict_stream(body: QuestionRequest) -> StreamingResponse:
    """
    Streams the answer token-by-token using Server-Sent Events (SSE).

    **Response format** — each line is a separate SSE event:
    ```
    data: {"delta": "Paris", "field": "answer"}
    data: {"delta": " is", "field": "answer"}
    ...
    data: {"answer": "Paris is the capital of France."}
    data: [DONE]
    ```
    Connect with `EventSource` in the browser or `httpx` with streaming in Python.
    """
    return StreamingResponse(
        _token_generator(body.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering when behind a proxy
        },
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "lm": os.getenv("LM_MODEL", "openai/gpt-4o-mini")}
