from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm
import WikiRequester

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)

TOP_K_PAGES = 5
TOP_K_TRAIN = 3
MAX_CONTEXT_TOKENS = 4000
FORGET_THRESHOLD = 6000
MIN_PAGES = 2
GPU_LAYERS = 0
CTX_SIZE = 16384

MODEL_URL = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

BASE_DIR = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
APP_DIR = os.path.join(BASE_DIR, "EUTGPT")
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.gguf")
DB_PATH = os.path.join(APP_DIR, "WikiDump.db")
TRAIN_PATH = os.path.join(APP_DIR, "train.jsonl")

os.makedirs(MODEL_DIR, exist_ok=True)


@dataclass(frozen=True)
class Page:
    title: str
    text: str
    tokens: Tuple[str, ...]


@dataclass(frozen=True)
class TrainExample:
    question: str
    answer: str
    pages: Tuple[str, ...]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_title(text: str) -> str:
    return normalize_text(text).lower()


def ensure_model(force_redownload: bool = False):
    if force_redownload and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        print("Model already exists.")
        return MODEL_PATH

    print("Downloading model (first run)...")

    with DownloadProgressBar(unit="B", unit_scale=True, desc="Model") as t:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=t.update_to)

    print("Model downloaded.")
    return MODEL_PATH


def load_llm(force_redownload: bool = False):
    ensure_model(force_redownload=force_redownload)

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
        verbose=False,
    )


class BM25PageRetriever:
    def __init__(self, pages: Sequence[Page], k1: float = 1.5, b: float = 0.75):
        self.pages = list(pages)
        self.k1 = k1
        self.b = b

        self.doc_len = [len(p.tokens) for p in pages]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

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
        return WORD_RE.findall((text or "").lower())

    def score(self, q, i):
        freq = self.doc_freqs[i]
        dl = self.doc_len[i]
        score = 0.0

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
        if not self.pages:
            return []

        q = self.tokenize(query)
        res = []

        for i, p in enumerate(self.pages):
            s = self.score(q, i)
            if s > 0:
                res.append((s, p))

        res.sort(reverse=True, key=lambda x: x[0])
        return res[:TOP_K_PAGES]


class BM25TrainRetriever:
    def __init__(self, examples: Sequence[TrainExample], k1: float = 1.5, b: float = 0.75):
        self.examples = list(examples)
        self.k1 = k1
        self.b = b

        self.doc_len = [len(self.tokenize(ex.question)) for ex in examples]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

        self.doc_freqs = []
        df = defaultdict(int)

        for ex in examples:
            tokens = self.tokenize(ex.question)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for t in freq:
                df[t] += 1

        self.df = dict(df)
        self.n_docs = len(examples)

    @staticmethod
    def tokenize(text: str):
        return WORD_RE.findall((text or "").lower())

    def score(self, q, i):
        freq = self.doc_freqs[i]
        dl = self.doc_len[i]
        score = 0.0

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
        if not self.examples:
            return []

        q = self.tokenize(query)
        res = []

        for i, ex in enumerate(self.examples):
            s = self.score(q, i)
            if s > 0:
                res.append((s, ex))

        res.sort(reverse=True, key=lambda x: x[0])
        return res[:TOP_K_TRAIN]


def load_pages(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    pages = []
    for row in conn.execute("SELECT title, content FROM pages"):
        title = normalize_text(row["title"])
        content = row["content"] or ""
        text = f"Title: {title}\n\n{content}"
        tokens = tuple(BM25PageRetriever.tokenize(text))
        pages.append(Page(title, text, tokens))

    conn.close()
    return pages


def build_page_lookup(pages: Sequence[Page]) -> Dict[str, Page]:
    lookup: Dict[str, Page] = {}
    for page in pages:
        lookup[normalize_title(page.title)] = page
    return lookup


def load_train_examples(train_path: str):
    if not os.path.exists(train_path):
        return []

    examples: List[TrainExample] = []

    with open(train_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = normalize_text(data.get("question", ""))
            answer = normalize_text(data.get("answer", ""))
            pages = data.get("pages", [])

            if not isinstance(pages, list):
                pages = []

            cleaned_pages = tuple(normalize_text(p) for p in pages if normalize_text(p))

            if question and answer:
                examples.append(TrainExample(question, answer, cleaned_pages))

    return examples


def append_train_example(train_path: str, question: str, answer: str, pages: Sequence[str]):
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    record = {
        "question": normalize_text(question),
        "answer": normalize_text(answer),
        "pages": [normalize_text(page) for page in pages if normalize_text(page)],
    }

    with open(train_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_context(results):
    out = []
    tokens = 0

    for _, p in results:
        chunk = f"[{p.title}]\n{p.text}"
        size = max(1, len(chunk) // 4)

        if tokens + size > MAX_CONTEXT_TOKENS and len(out) >= MIN_PAGES:
            break

        out.append(chunk)
        tokens += size

    return "\n\n".join(out)


def build_train_context(results, page_lookup: Dict[str, Page]):
    if not results:
        return ""

    out = ["TRAINED EXAMPLES: Use these as strongly relevant memory when they match the question."]

    for score, ex in results[:TOP_K_TRAIN]:
        block = [
            f"[Train Match | score={score:.3f}]",
            f"Question: {ex.question}",
            f"Answer: {ex.answer}",
        ]

        related_pages = []
        for page_name in ex.pages:
            page = page_lookup.get(normalize_title(page_name))
            if page:
                related_pages.append(f"[{page.title}]\n{page.text}")

        if related_pages:
            block.append("Relevant Wiki Pages:\n" + "\n\n".join(related_pages))

        out.append("\n".join(block))

    return "\n\n".join(out)


async def ensure_database(force_rebuild: bool = False):
    if force_rebuild and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    if os.path.exists(DB_PATH):
        return

    print("Generating DB...")
    await WikiRequester.main()
    print("DB ready.")


def stream(llm, prompt):
    print("Assistant:", end=" ")

    out = ""
    for c in llm.create_completion(prompt, stream=True, max_tokens=1200):
        t = c["choices"][0].get("text", "")
        out += t
        print(t, end="", flush=True)

    print()
    return out


def build_reasoning_prompt(context: str, question: str, history: str) -> str:
    return f"""
You are a strict wiki reasoning engine.

IMPORTANT:
- This is a NEW and INDEPENDENT question.
- Do NOT reuse reasoning from previous questions.
- Keep reasoning concise and focused on correctness.
- You must primarily REWRITE information in your own words.

COPYING RULE:
- You are NOT allowed to copy text verbatim.
- EXCEPTION: You may copy ONLY when directly quoting a source from CONTEXT.
- If you copy, it must be clearly marked as a quotation.
- Otherwise, everything must be paraphrased and rewritten.

CHAT HISTORY:
{history}

CONTEXT:
{context}

QUESTION:
{question}

OUTPUT RULES:
- Prefer rewriting over repetition
- Combine related facts instead of repeating them
- Do not mirror sentence structure from context unless quoting

OUTPUT FORMAT:
Extracted Facts:
Computation:
Validation:
Re-validation:
Final Answer:
"""


async def main():
    await ensure_database()

    pages = load_pages(DB_PATH)
    page_lookup = build_page_lookup(pages)
    bm25 = BM25PageRetriever(pages) if pages else None

    train_examples = load_train_examples(TRAIN_PATH)
    train_bm25 = BM25TrainRetriever(train_examples) if train_examples else None

    llm = load_llm()

    print("System ready")
    print("Type Chat or Train.")
    print("Commands: /updatewiki, /reloadmodel, /train, /back, /quit, /endtrain")

    history = []
    mode = ""

    while mode not in {"chat", "train"}:
        mode = normalize_text(input("Mode (Chat/Train): ")).lower()

        if mode in {"/quit", "/exit"}:
            return

        if mode == "/train":
            mode = "train"

        elif mode == "/updatewiki":
            await ensure_database(force_rebuild=True)
            pages = load_pages(DB_PATH)
            page_lookup = build_page_lookup(pages)
            bm25 = BM25PageRetriever(pages) if pages else None
            print("Wiki database updated.")
            mode = ""

        elif mode == "/reloadmodel":
            llm = load_llm(force_redownload=True)
            print("Model reloaded.")
            mode = ""

    while True:
        if mode == "chat":
            q = normalize_text(input("You: "))
            if not q:
                continue

            lower = q.lower()

            if lower in {"/quit", "/exit"}:
                break

            if lower == "/train":
                mode = "train"
                continue

            if lower == "/back":
                mode = ""
                break

            if lower == "/updatewiki":
                await ensure_database(force_rebuild=True)
                pages = load_pages(DB_PATH)
                page_lookup = build_page_lookup(pages)
                bm25 = BM25PageRetriever(pages) if pages else None
                print("Wiki database updated.")
                continue

            if lower == "/reloadmodel":
                llm = load_llm(force_redownload=True)
                print("Model reloaded.")
                continue

            results = bm25.search(q) if bm25 else []
            train_results = train_bm25.search(q) if train_bm25 else []

            wiki_context = build_context(results)
            train_context = build_train_context(train_results, page_lookup)

            context = train_context + "\n\n" + wiki_context if train_context and wiki_context else train_context or wiki_context

            history_text = ""
            for u, a in history:
                history_text += f"User: {u}\nAssistant: {a}\n"

            prompt = build_reasoning_prompt(context, q, history_text)

            if llm:
                answer = stream(llm, prompt)
            else:
                answer = results[0][1].text if results else "No relevant wiki pages found."
                print(answer)

            history.append((q, answer[:200]))

        else:
            question = normalize_text(input("Question: "))
            lower = question.lower()

            if not question:
                continue

            if lower in {"/quit", "/exit"}:
                return

            if lower == "/back" or lower == "/endtrain":
                mode = ""
                break

            if lower == "/updatewiki":
                await ensure_database(force_rebuild=True)
                pages = load_pages(DB_PATH)
                page_lookup = build_page_lookup(pages)
                bm25 = BM25PageRetriever(pages) if pages else None
                print("Wiki database updated.")
                continue

            if lower == "/reloadmodel":
                llm = load_llm(force_redownload=True)
                print("Model reloaded.")
                continue

            answer = normalize_text(input("Answer: "))
            pages_to_save: List[str] = []
            page_num = 1

            while True:
                page_name = normalize_text(input(f"Page {page_num}: "))
                if page_name.lower() == "/endtrain":
                    break
                if page_name:
                    pages_to_save.append(page_name)
                    page_num += 1

            append_train_example(TRAIN_PATH, question, answer, pages_to_save)

            train_examples = load_train_examples(TRAIN_PATH)
            train_bm25 = BM25TrainRetriever(train_examples) if train_examples else None

            print("Updating train file\n")


if __name__ == "__main__":
    asyncio.run(main())