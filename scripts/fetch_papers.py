"""
Download the sample research-paper corpus into data/documents/ from arXiv.

The corpus is deliberately self-referential: it's made up of the papers underlying this
project's own RAG pipeline (transformers, RAG itself, sentence-transformers, dense
retrieval), plus two more covering other ML areas, so the demo can be asked to explain
the exact techniques it's built from. PDFs are fetched by arXiv ID rather than committed
to the repo -- arXiv preprints are freely accessible, and this keeps the repo free of
multi-megabyte binaries while making data acquisition a reproducible, visible step.

Safe to re-run: files that already exist are left alone.

Runnable as: python -m scripts.fetch_papers
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"

# (arXiv ID, output filename)
PAPERS = [
    ("1706.03762", "vaswani_2017_attention_is_all_you_need.pdf"),
    ("2005.11401", "lewis_2020_retrieval_augmented_generation.pdf"),
    ("1908.10084", "reimers_2019_sentence_bert.pdf"),
    ("2004.04906", "karpukhin_2020_dense_passage_retrieval.pdf"),
    ("2010.11929", "dosovitskiy_2020_vision_transformer.pdf"),
    ("2210.03629", "yao_2022_react.pdf"),
]

USER_AGENT = "portfolio-rag-qa/1.0 (fetch_papers.py; educational use)"


def fetch(arxiv_id: str, filename: str) -> None:
    dest = DATA_DIR / filename
    if dest.exists():
        print(f"  - {filename}: already present, skipping")
        return

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"  - {filename}: downloaded ({len(response.content) // 1024} KB)")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(PAPERS)} paper(s) from arXiv into {DATA_DIR}")
    for arxiv_id, filename in PAPERS:
        fetch(arxiv_id, filename)


if __name__ == "__main__":
    main()
