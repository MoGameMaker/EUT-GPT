import aiohttp
import asyncio
import sqlite3
import re
import time
import os

from paths import DB_PATH  # IMPORTANT: persistent path

API_URL = "https://eutwiki.com/w/api.php"
HEADERS = {"User-Agent": "WikiIndexer/4.0 (Python)"}

BATCH_SIZE = 50


# --- WIKITEXT CLEANER ---
def wikitext_to_plain(text: str) -> str:
    if not text:
        return ""

    if text.strip().lower().startswith("#redirect"):
        return ""

    # tables (basic cleanup)
    text = re.sub(r"^\s*\{\|.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|\}.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|\-.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\!", " ", text, flags=re.MULTILINE)

    # templates
    for _ in range(6):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # links
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    # external links
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)

    # headings
    text = re.sub(r"={2,6}\s*(.+?)\s*={2,6}", r"\1", text)

    # formatting
    text = re.sub(r"'{2,3}", "", text)

    # html
    text = re.sub(r"<[^>]+>", "", text)

    # cleanup
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --- DB SETUP ---
def init_db(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS pages (
        title TEXT PRIMARY KEY,
        content TEXT
    );

    CREATE TABLE IF NOT EXISTS page_links (
        from_title TEXT,
        to_title TEXT
    );

    CREATE TABLE IF NOT EXISTS redirects (
        alias TEXT PRIMARY KEY,
        target TEXT
    );
    """)
    conn.commit()


# --- FETCH ---
async def fetch_all(session):
    pages = []
    links = []
    redirects = []
    cont = {}

    batch = 0

    while True:
        batch += 1

        params = {
            "action": "query",
            "generator": "allpages",
            "gaplimit": str(BATCH_SIZE),
            "gapnamespace": "0",
            "prop": "revisions|links|redirects",
            "rvprop": "content",
            "rvslots": "main",
            "pllimit": "max",
            "rdlimit": "max",
            "format": "json",
            "formatversion": "2",
        }

        params.update(cont)

        try:
            async with session.get(API_URL, params=params, timeout=30) as r:
                data = await r.json()
        except Exception as e:
            print(f"\nBatch {batch} failed: {e}")
            break

        for page in data.get("query", {}).get("pages", []):
            if page.get("missing"):
                continue

            title = page["title"]

            wikitext = ""
            for rev in page.get("revisions", []):
                slots = rev.get("slots", {})
                wikitext = slots.get("main", {}).get("content", "") or rev.get("content", "")
                break

            content = wikitext_to_plain(wikitext)

            if content:
                pages.append((title, content))

            for l in page.get("links", []):
                links.append((title, l["title"]))

            for r in page.get("redirects", []):
                redirects.append((r["title"], title))

        print(f"Batch {batch}: {len(pages)} pages", end="\r")

        if "continue" not in data:
            break

        cont = data["continue"]

    return pages, links, redirects


# --- MAIN ---
async def main():
    start = time.perf_counter()

    print("WikiRequester starting...")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        pages, links, redirects = await fetch_all(session)

    print(f"\nFetched {len(pages)} pages")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # atomic-ish insert (prevents corruption issues)
    conn.executemany(
        "INSERT OR REPLACE INTO pages VALUES (?, ?)",
        pages
    )

    conn.executemany(
        "INSERT INTO page_links VALUES (?, ?)",
        links
    )

    conn.executemany(
        "INSERT OR REPLACE INTO redirects VALUES (?, ?)",
        redirects
    )

    conn.commit()
    conn.close()

    print(f"DB saved → {DB_PATH}")
    print(f"Done in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())