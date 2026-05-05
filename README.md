# DSPy + FastAPI — Normal & Streaming Responses

A minimal but production-ready reference implementation that shows how to build two kinds of AI inference endpoints with **DSPy 3.x** and **FastAPI**:

| Endpoint | Mode | Use when |
|---|---|---|
| `POST /predict` | Normal (blocking JSON) | You need the full answer before doing anything else |
| `POST /predict/stream` | Streaming SSE | You want to show the answer to a user token-by-token |

---

## Table of Contents

1. [What is DSPy?](#what-is-dspy)
2. [Project structure](#project-structure)
3. [How it works](#how-it-works)
   - [DSPy Signature](#dspy-signature)
   - [DSPy Module](#dspy-module)
   - [Normal endpoint](#normal-endpoint)
   - [Streaming endpoint](#streaming-endpoint)
4. [Prerequisites](#prerequisites)
5. [Setup](#setup)
6. [Running the server](#running-the-server)
7. [Testing the endpoints](#testing-the-endpoints)
   - [Using the test client](#using-the-test-client)
   - [Using curl](#using-curl)
   - [Using the interactive docs](#using-the-interactive-docs)
   - [Using EventSource in a browser](#using-eventsource-in-a-browser)
8. [Supported LLM providers](#supported-llm-providers)
9. [API reference](#api-reference)
10. [Streaming response format explained](#streaming-response-format-explained)

---

## What is DSPy?

[DSPy](https://github.com/stanfordnlp/dspy) is a Stanford framework for **programming** (not just prompting) language models. Instead of writing and tweaking prompt strings by hand, you:

1. Declare a typed **Signature** — inputs and outputs with docstring instructions.
2. Build a **Module** that chains `Predict`, `ChainOfThought`, `ReAct`, or other built-in predictors.
3. Optionally **compile** (optimize) the module using a DSPy optimizer and a small labelled dataset — the framework rewrites prompts and few-shot examples automatically.

This project uses DSPy as the inference layer and FastAPI as the HTTP layer on top.

---

## Project structure

```
DSPY-Complete/
├── main.py           # FastAPI app — all endpoints live here
├── test_client.py    # Async Python client that exercises both endpoints
├── .env.example      # Template for environment variables — copy to .env
└── myenv/            # Python virtual environment (created during setup)
```

---

## How it works

### DSPy Signature

```python
class QASignature(dspy.Signature):
    """Answer the question thoroughly and clearly."""

    question: str = dspy.InputField()
    answer: str   = dspy.OutputField()
```

A `Signature` is a typed contract. The docstring becomes the task instruction sent to the LLM. `InputField` and `OutputField` tell DSPy which variables to pass in and which to parse out of the model's response.

### DSPy Module

```python
class QAModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(QASignature)

    def forward(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)
```

`dspy.Predict` is the simplest predictor — it formats the signature into a prompt, calls the LLM, and parses the output back into a `dspy.Prediction` object. The `forward` method is what gets called when you invoke the module like a function: `qa_module(question="...")`.

### Normal endpoint

```
POST /predict
```

The module is called once and waits for the LLM to finish generating the complete answer. The full text is returned as a single JSON response. Simple and appropriate when the caller can tolerate the wait (background jobs, batch processing, internal API calls).

```
Client  ──POST /predict──►  FastAPI  ──dspy.Predict──►  LLM
        ◄──── JSON ─────────────────────────────────────────
```

### Streaming endpoint

```
POST /predict/stream
```

The same `QAModule` is wrapped with `dspy.streamify()`, which converts it into an async generator. As the LLM produces tokens, each token is immediately forwarded to the client via **Server-Sent Events (SSE)** — no waiting for the full response.

```
Client  ──POST /predict/stream──►  FastAPI  ──dspy.streamify──►  LLM
        ◄── token ──────────────────────────────────────────────────
        ◄── token ──────────────────────────────────────────────────
        ◄── [DONE] ─────────────────────────────────────────────────
```

SSE is a standard HTTP mechanism (not WebSocket). The connection is a normal long-lived HTTP response with `Content-Type: text/event-stream`. Each event is a line that starts with `data:`.

The async generator `_token_generator` handles three kinds of objects yielded by DSPy:

| DSPy yields | Meaning | SSE event sent |
|---|---|---|
| `StreamResponse` | One token delta | `data: {"delta": "...", "field": "answer"}` |
| `dspy.Prediction` | Final assembled answer | `data: {"answer": "..."}` |
| *(generator exhausted)* | Stream done | `data: [DONE]` |

---

## Prerequisites

- Python 3.10 or later
- An API key for a supported LLM provider (OpenAI, Anthropic, etc.)

---

## Setup

**1. Create and activate the virtual environment**

```bash
python3 -m venv myenv
source myenv/bin/activate        # macOS / Linux
# myenv\Scripts\activate         # Windows
```

**2. Install dependencies**

```bash
pip install dspy fastapi uvicorn httpx python-dotenv
```

**3. Configure your API key**

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```env
OPENAI_API_KEY=sk-...
LM_MODEL=openai/gpt-4o-mini
```

See [Supported LLM providers](#supported-llm-providers) for other options.

---

## Running the server

```bash
# development — auto-reloads on file changes
uvicorn main:app --reload

# production — multiple workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The server starts at `http://127.0.0.1:8000`.

---

## Testing the endpoints

### Using the test client

With the server running, open a second terminal and run:

```bash
python test_client.py
```

Expected output:

```
============================================================
NORMAL endpoint  POST /predict
============================================================
Question : What is the capital of France and why is it historically significant?
Answer   : Paris is the capital of France. It has been the country's political...

============================================================
STREAMING endpoint  POST /predict/stream  (SSE)
============================================================
Tokens  : Paris is the capital of France. It has been...
Final   : Paris is the capital of France. It has been the country's political...
Stream finished.
```

### Using curl

**Normal endpoint:**

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the speed of light?"}' | python3 -m json.tool
```

**Streaming endpoint:**

```bash
curl -N -X POST http://127.0.0.1:8000/predict/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the speed of light?"}'
```

The `-N` flag disables curl's output buffering so you see tokens as they arrive.

### Using the interactive docs

FastAPI generates Swagger UI automatically. Open your browser at:

```
http://127.0.0.1:8000/docs
```

You can try both endpoints directly from the browser without writing any code.

### Using EventSource in a browser

For frontend applications, use the browser's native `EventSource` API. Because `EventSource` only supports `GET`, use `fetch` with streaming for `POST` endpoints:

```js
const response = await fetch("http://127.0.0.1:8000/predict/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "What is the speed of light?" }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (!line.startsWith("data:")) continue;
    const payload = line.slice(5).trim();
    if (payload === "[DONE]") break;

    const data = JSON.parse(payload);
    if (data.delta) {
      // append token to UI
      document.getElementById("output").textContent += data.delta;
    }
  }
}
```

---

## Supported LLM providers

DSPy uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, so any provider LiteLLM supports works here. Set `LM_MODEL` and the corresponding API key in your `.env`.

| Provider | `LM_MODEL` value | Key variable |
|---|---|---|
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-3-5-haiku-latest` | `ANTHROPIC_API_KEY` |
| Anthropic | `anthropic/claude-opus-4-7` | `ANTHROPIC_API_KEY` |
| Google | `gemini/gemini-1.5-flash` | `GEMINI_API_KEY` |
| Groq | `groq/llama3-8b-8192` | `GROQ_API_KEY` |
| Ollama (local) | `ollama/llama3` | *(no key needed)* |

---

## API reference

### `GET /health`

Returns server status and the configured model name.

**Response:**
```json
{"status": "ok", "lm": "openai/gpt-4o-mini"}
```

---

### `POST /predict`

Returns the complete answer once generation finishes.

**Request body:**
```json
{"question": "Your question here"}
```

**Response `200 OK`:**
```json
{
  "question": "Your question here",
  "answer": "The complete answer from the LLM."
}
```

**Error `500`:**
```json
{"detail": "error message from the LLM or DSPy"}
```

---

### `POST /predict/stream`

Streams the answer token-by-token as Server-Sent Events.

**Request body:**
```json
{"question": "Your question here"}
```

**Response `200 OK`** — `Content-Type: text/event-stream`

Each line of the response body is one SSE event:

```
data: {"delta": "The", "field": "answer"}
data: {"delta": " complete", "field": "answer"}
data: {"delta": " answer", "field": "answer"}
data: {"delta": ".", "field": "answer"}
data: {"answer": "The complete answer."}
data: [DONE]
```

| Field | Present in | Description |
|---|---|---|
| `delta` | Token events | The new token/chunk just generated |
| `field` | Token events | The DSPy output field this token belongs to (`"answer"`) |
| `answer` | Final event | The full assembled answer string |

---

## Streaming response format explained

SSE is a simple line-based protocol over HTTP:

```
data: <payload>\n\n
```

- Each event ends with **two newlines** (`\n\n`).
- The `data:` prefix is the SSE field name — browsers parse this automatically.
- `[DONE]` is a sentinel value (borrowed from OpenAI's convention) that tells the client the stream is finished.

This format is natively understood by browsers (`EventSource`), `httpx`, `requests` (with streaming), `curl -N`, and most HTTP client libraries.
# DSPYStreaming
