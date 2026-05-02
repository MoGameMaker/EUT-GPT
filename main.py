from __future__ import annotations

import math
import os
import re
import sqlite3
import sys
import urllib.request
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from tqdm import tqdm
import WikiRequester

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)

TOP_K_PAGES = 5
MAX_CONTEXT_TOKENS = 4000
FORGET_THRESHOLD = 6000
MIN_PAGES = 2
GPU_LAYERS = 0
CTX_SIZE = 16384

MODEL_URL = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"


# ----------------------------
# APP DATA PATH (IMPORTANT FIX)
# ----------------------------
APP_DIR = os.path.join(os.getenv("LOCALAPPDATA"), "EUTGPT")
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.gguf")
DB_PATH = os.path.join(APP_DIR, "WikiDump.db")

os.makedirs(MODEL_DIR, exist_ok=True)


@dataclass(frozen=True)
class Page:
    title: str
    text: str
    tokens: Tuple[str, ...]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# ----------------------------
# MODEL HANDLING (FIXED)
# ----------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        print("Model already exists.")
        return MODEL_PATH

    print("Downloading model (first run)...")

    with DownloadProgressBar(unit="B", unit_scale=True, desc="Model") as t:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=t.update_to)

    print("Model downloaded.")
    return MODEL_PATH


def load_llm(_path=None):
    ensure_model()

    try:
        from llama_cpp import Llama
    except Exception as e:
        print("llama_cpp import failed:", repr(e))
        return None

    return Llama(
        model_path=MODEL_PATH,
        n_ctx=CTX_SIZE,
        n_gpu_layers=GPU_LAYERS,
        n_threads=os.cpu_count(),
        n_batch=512,
        verbose=False
    )


# ----------------------------
# BM25 (UNCHANGED CORE LOGIC)
# ----------------------------
class BM25PageRetriever:
    def __init__(self, pages: Sequence[Page], k1: float = 1.5, b: float = 0.75):
        self.pages = list(pages)
        self.k1 = k1
        self.b = b

        self.doc_len = [len(p.tokens) for p in pages]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)

        self.doc_freqs = []
        df = defaultdict(int)

        for p in pages:
            freq = Counter(p.tokens)
            self.doc_freqs.append(freq)
            for t in freq:
                df[t] += 1

        self.df = dict(df)
        self.n_docs = len(pages)

    @staticmethod
    def tokenize(text: str):
        return WORD_RE.findall(text.lower())

    def score(self, q, i):
        freq = self.doc_freqs[i]
        dl = self.doc_len[i]
        score = 0

        for term in q:
            tf = freq.get(term, 0)
            if not tf:
                continue

            df = self.df.get(term, 0)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

            denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
            score += idf * (tf * (self.k1 + 1)) / denom

        return score

    def search(self, query):
        q = self.tokenize(query)
        res = []

        for i, p in enumerate(self.pages):
            s = self.score(q, i)
            if s > 0:
                res.append((s, p))

        res.sort(reverse=True, key=lambda x: x[0])
        return res[:TOP_K_PAGES]


# ----------------------------
# DB LOADING (FIXED PATH)
# ----------------------------
def load_pages(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    pages = []
    for row in conn.execute("SELECT title, content FROM pages"):
        title = re.sub(r"\s+", " ", row["title"]).strip()
        content = row["content"] or ""

        text = f"Title: {title}\n\n{content}"
        tokens = tuple(BM25PageRetriever.tokenize(text))

        pages.append(Page(title, text, tokens))

    return pages


# ----------------------------
def build_context(results):
    out = []
    tokens = 0

    for _, p in results:
        chunk = f"[{p.title}]\n{p.text}"
        size = len(chunk) // 4

        if tokens + size > MAX_CONTEXT_TOKENS and len(out) >= MIN_PAGES:
            break

        out.append(chunk)
        tokens += size

    return "\n\n".join(out)


# ----------------------------
async def ensure_database():
    if os.path.exists(DB_PATH):
        return

    print("Generating DB...")
    await WikiRequester.main()
    print("DB ready.")


# ----------------------------
def stream(llm, prompt):
    print("Assistant:", end=" ")

    out = ""
    for c in llm.create_completion(prompt, stream=True, max_tokens=1200):
        t = c["choices"][0].get("text", "")
        out += t
        print(t, end="", flush=True)

    print()
    return out


# ----------------------------
# YOUR SYSTEM PROMPT (UNCHANGED)
# ----------------------------
def build_reasoning_prompt(context: str, question: str, history: str) -> str:
    return f"""
You are a strict wiki reasoning engine.

IMPORTANT:
- This is a NEW and INDEPENDENT question.
- Do NOT reuse reasoning from previous questions.
- Keep reasoning concise. Prioritize completing the Final Answer FULLY over lengthy reasonining although this doesn't mean you should skip reasoning. Just be concise and to the point.

You MUST follow this pipeline:
1. Extract ONLY relevant facts from context
2. Compute required values (if any)
3. Validate result against context
4. Re-validate consistency
5. Output final answer

Rules:
- Do NOT copy raw context
- Do NOT assume previous questions are related
- If missing info, say so

CHAT HISTORY:
{history}

CONTEXT:
{context}

QUESTION:
{question}

OUTPUT FORMAT:
Extracted Facts:
Computation:
Validation:
Re-validation:
Final Answer:
"""


# ----------------------------
async def main():
    await ensure_database()

    pages = load_pages(DB_PATH)
    bm25 = BM25PageRetriever(pages)
    llm = load_llm()

    print("System ready")

    history = []

    while True:
        q = input("You: ").strip()
        if q in {"/exit", "/quit"}:
            break

        results = bm25.search(q)
        context = build_context(results)

        history_text = ""
        for u, a in history:
            history_text += f"User: {u}\nAssistant: {a}\n"

        prompt = build_reasoning_prompt(context, q, history_text)

        if llm:
            answer = stream(llm, prompt)
        else:
            answer = results[0][1].text
            print(answer)

        history.append((q, answer[:200]))


if __name__ == "__main__":
    asyncio.run(main())