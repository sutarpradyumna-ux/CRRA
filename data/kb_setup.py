"""
Build a ChromaDB knowledge base from markdown IT articles and test it.

1. Reads every .md file in data/kb/
2. Splits each article into chunks at '## ' headings
3. Stores all chunks in the ChromaDB collection 'isdo_kb'
4. Runs 6 sample queries and prints the best-matching article + confidence

Requirements:  pip install chromadb
Run:           python build_kb.py
"""

import re
from pathlib import Path

import chromadb

KB_DIR = Path("data/kb")
DB_DIR = "chroma_db"            # persistent storage folder
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "I forgot my password and my account is locked",
    "VPN keeps disconnecting when I work from home",
    "My laptop is running very slow",
    "Outlook is not receiving new emails",

    # Break test 1: the policy does not cover this, so expect a LOW confidence score
    "What is the travel expense limit?",

    # Break test 2: the answer spans two articles (contract value + termination)
    "Can I terminate a 90 lakh contract early?",
]


def split_markdown(text: str, source: str) -> list[dict]:
    """Split one markdown article into chunks at '## ' headings."""
    # Article title = first '# ' heading, else the file name
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source

    # Split right before every line that starts with '## ' (not '###')
    parts = re.split(r"(?m)^(?=##\s)", text)

    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        heading_match = re.match(r"##\s+(.+)", part)
        section = heading_match.group(1).strip() if heading_match else "Introduction"

        # Skip an intro chunk that is only the '# Title' line
        body = re.sub(r"^#\s+.+$", "", part, flags=re.MULTILINE).strip()
        if not body:
            continue

        chunks.append({
            "id": f"{source}::{i}",
            # Prefix with the title so every chunk carries article context
            "text": f"{title}\n\n{part}",
            "metadata": {"article": source, "title": title, "section": section},
        })
    return chunks


def load_chunks(kb_dir: Path) -> list[dict]:
    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {kb_dir.resolve()}")

    all_chunks = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        chunks = split_markdown(text, path.name)
        print(f"  {path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


def build_collection(chunks: list[dict]):
    client = chromadb.PersistentClient(path=DB_DIR)

    # Start fresh on every run so re-running doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Cosine distance lets us report confidence as 1 - distance
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection


def run_queries(collection, queries: list[str]):
    print("\n" + "=" * 70)
    print("SAMPLE QUERIES")
    print("=" * 70)
    for q in queries:
        res = collection.query(query_texts=[q], n_results=1)
        meta = res["metadatas"][0][0]
        distance = res["distances"][0][0]
        confidence = max(0.0, 1 - distance)

        print(f"\nQuery:      {q}")
        print(f"Article:    {meta['article']}  ({meta['title']})")
        print(f"Section:    {meta['section']}")
        print(f"Confidence: {confidence:.2%}")


def main():
    print(f"Reading articles from {KB_DIR}/ ...")
    chunks = load_chunks(KB_DIR)

    print(f"\nStoring {len(chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}' ...")
    collection = build_collection(chunks)
    print(f"Collection now holds {collection.count()} chunks.")

    run_queries(collection, SAMPLE_QUERIES)


if __name__ == "__main__":
    main()