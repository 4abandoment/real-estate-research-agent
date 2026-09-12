"""Policy / lease knowledge base from real UK government guidance.

Fetches public guidance (Open Government Licence v3.0), converts HTML to text,
chunks and embeds it so policy questions can be answered with citations.
"""

import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from research_agent.config import load_settings
from research_agent.db import apply_schema, connect
from research_agent.embeddings import Embedder

PAGES = [
    ("https://www.gov.uk/renting-out-a-property", "Renting out a property"),
    ("https://www.gov.uk/tenancy-deposit-protection", "Tenancy deposit protection"),
    ("https://www.gov.uk/evicting-tenants", "Evicting tenants"),
    ("https://www.gov.uk/rent-room-in-your-home", "Renting a room in your home"),
]
RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "policy"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs) -> None:
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data) -> None:
        if not self._skip and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts))


def fetch_text(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_text(encoding="utf-8")
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        html = response.read().decode("utf-8", "ignore")
    parser = _TextExtractor()
    parser.feed(html)
    text = parser.text()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return text


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [item for item in chunks if len(item.strip()) > 50]


def load_policy(conn, embedder: Embedder, pages: list[tuple[str, str]] = PAGES) -> int:
    conn.execute("TRUNCATE policy_documents")
    total = 0
    with conn.cursor() as cursor:
        for url, title in pages:
            slug = url.rstrip("/").split("/")[-1]
            try:
                text = fetch_text(url, RAW / f"{slug}.txt")
            except Exception as error:  # noqa: BLE001 - one bad page must not stop the load
                print(f"skipping {url}: {error}")
                continue
            chunks = chunk(text)
            vectors = embedder.embed(chunks)
            for index, (content, vector) in enumerate(zip(chunks, vectors, strict=True)):
                cursor.execute(
                    "INSERT INTO policy_documents"
                    " (source_url, title, chunk_index, content, embedding)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    (url, title, index, content, Embedder.to_pgvector(vector)),
                )
                total += 1
    return total


def main() -> None:
    database_url = load_settings().database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required.")
    conn = connect(database_url)
    apply_schema(conn)
    print(f"policy chunks: {load_policy(conn, Embedder())}")


if __name__ == "__main__":
    main()
