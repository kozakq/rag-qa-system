"""
Retrieval + answer quality evaluation against eval/questions.json.

This is the module that proves the system actually works, not just that it runs.
A resume line like "Model Evaluation" needs a real report behind it — this produces one.

- Load eval/questions.json: [{"question": str, "expected_source": str, "expected_keywords": [str]}]
- For each question: retrieve top_k chunks, check whether expected_source appears among them
  (retrieval accuracy), then generate an answer and check whether expected_keywords appear in
  it (rough answer-quality proxy — good enough for a portfolio project, no need for anything
  more elaborate)
- Print/save a small report: retrieval accuracy %, answer-keyword-hit %, and per-question detail
- Should be runnable as: python -m scripts.run_eval
"""

import json
from pathlib import Path

from src import generate, retrieve

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "questions.json"


def evaluate(questions_path: Path = QUESTIONS_PATH, top_k: int = 4) -> dict:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    details = []
    retrieval_hits = 0
    answer_hits = 0

    for item in questions:
        chunks = retrieve.retrieve(item["question"], top_k=top_k)
        sources = [c["source"] for c in chunks]
        retrieval_hit = item["expected_source"] in sources

        context = retrieve.format_context(chunks)
        answer = generate.generate_answer(item["question"], context)

        matched_keywords = [
            kw for kw in item["expected_keywords"] if kw.lower() in answer.lower()
        ]
        answer_hit = len(matched_keywords) > 0

        if retrieval_hit:
            retrieval_hits += 1
        if answer_hit:
            answer_hits += 1

        details.append(
            {
                "question": item["question"],
                "expected_source": item["expected_source"],
                "retrieved_sources": sources,
                "retrieval_hit": retrieval_hit,
                "answer": answer,
                "expected_keywords": item["expected_keywords"],
                "matched_keywords": matched_keywords,
                "answer_hit": answer_hit,
            }
        )

    total = len(questions)
    retrieval_accuracy = (retrieval_hits / total * 100) if total else 0.0
    answer_keyword_hit_rate = (answer_hits / total * 100) if total else 0.0

    report = {
        "total_questions": total,
        "retrieval_accuracy": retrieval_accuracy,
        "answer_keyword_hit_rate": answer_keyword_hit_rate,
        "details": details,
    }

    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    print("=" * 60)
    print("RAG Evaluation Report")
    print("=" * 60)
    for i, d in enumerate(report["details"], start=1):
        print(f"\n[{i}] {d['question']}")
        print(f"    expected source: {d['expected_source']}")
        print(f"    retrieved sources: {d['retrieved_sources']}")
        print(f"    retrieval hit: {'YES' if d['retrieval_hit'] else 'NO'}")
        print(f"    answer: {d['answer']}")
        print(
            f"    keyword hit: {'YES' if d['answer_hit'] else 'NO'} "
            f"(matched {d['matched_keywords']} of {d['expected_keywords']})"
        )
    print("\n" + "=" * 60)
    print(f"Total questions:          {report['total_questions']}")
    print(f"Retrieval accuracy:       {report['retrieval_accuracy']:.1f}%")
    print(f"Answer keyword-hit rate:  {report['answer_keyword_hit_rate']:.1f}%")
    print("=" * 60)
