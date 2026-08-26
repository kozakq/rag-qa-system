"""
Retrieval + answer quality evaluation against eval/questions.json.

This is the module that proves the system actually works, not just that it runs.
A resume line like "Model Evaluation" needs a real report behind it — this produces one.

- Load eval/questions.json: a list of question objects. Most are answerable:
  {"question": str, "expected_source": str, "expected_keywords": [str]}. Some are
  deliberately unanswerable (nothing in the corpus covers them): {"question": str,
  "answerable": false} — these test that the system says "I don't know" instead of
  falling back on the model's own training data, which is the actual point of RAG.
- For each answerable question: retrieve top_k chunks, check whether expected_source
  appears among them (retrieval accuracy) and whether it's specifically the #1 result
  (a stricter top-1 accuracy — the lenient top_k check can't fail once the corpus is
  small relative to top_k, so this is what actually catches ranking/pollution issues),
  then generate an answer and check whether expected_keywords appear in it (rough
  answer-quality proxy — good enough for a portfolio project, no need for anything
  more elaborate).
- For each unanswerable question: generate an answer anyway and check whether it reads
  as a refusal (a small set of "I don't know"-style substrings) rather than a
  confident-sounding guess.
- Print/save a small report: retrieval accuracy %, top-1 retrieval accuracy %,
  answer-keyword-hit %, refusal accuracy %, and per-question detail.
- Should be runnable as: python -m scripts.run_eval
"""

import json
from pathlib import Path

from src import generate, retrieve

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "questions.json"

REFUSAL_MARKERS = [
    "i don't know",
    "i do not know",
    "cannot find",
    "can't find",
    "does not mention",
    "doesn't mention",
    "not contain",
    "no information",
    "not provided",
    "does not provide",
    "not mentioned",
    "unable to find",
]


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _evaluate_answerable(item: dict, chunks: list[dict], answer: str, top_k: int) -> dict:
    sources = [c["source"] for c in chunks]
    retrieval_hit = item["expected_source"] in sources

    top1 = retrieve.retrieve(item["question"], top_k=1)
    retrieval_hit_top1 = bool(top1) and top1[0]["source"] == item["expected_source"]

    matched_keywords = [kw for kw in item["expected_keywords"] if kw.lower() in answer.lower()]
    answer_hit = len(matched_keywords) > 0

    return {
        "answerable": True,
        "question": item["question"],
        "expected_source": item["expected_source"],
        "retrieved_sources": sources,
        "retrieval_hit": retrieval_hit,
        "retrieval_hit_top1": retrieval_hit_top1,
        "answer": answer,
        "expected_keywords": item["expected_keywords"],
        "matched_keywords": matched_keywords,
        "answer_hit": answer_hit,
    }


def _evaluate_unanswerable(item: dict, chunks: list[dict], answer: str) -> dict:
    return {
        "answerable": False,
        "question": item["question"],
        "retrieved_sources": [c["source"] for c in chunks],
        "answer": answer,
        "refused": _looks_like_refusal(answer),
    }


def evaluate(questions_path: Path = QUESTIONS_PATH, top_k: int = 4) -> dict:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    details = []
    for item in questions:
        chunks = retrieve.retrieve(item["question"], top_k=top_k)
        context = retrieve.format_context(chunks)
        answer = generate.generate_answer(item["question"], context)

        if item.get("answerable", True):
            details.append(_evaluate_answerable(item, chunks, answer, top_k))
        else:
            details.append(_evaluate_unanswerable(item, chunks, answer))

    answerable = [d for d in details if d["answerable"]]
    unanswerable = [d for d in details if not d["answerable"]]

    def rate(hits: int, total: int) -> float | None:
        return (hits / total * 100) if total else None

    report = {
        "total_questions": len(questions),
        "top_k": top_k,
        "retrieval_accuracy": rate(sum(d["retrieval_hit"] for d in answerable), len(answerable)),
        "retrieval_accuracy_top1": rate(
            sum(d["retrieval_hit_top1"] for d in answerable), len(answerable)
        ),
        "answer_keyword_hit_rate": rate(sum(d["answer_hit"] for d in answerable), len(answerable)),
        "refusal_accuracy": rate(sum(d["refused"] for d in unanswerable), len(unanswerable)),
        "details": details,
    }

    _print_report(report)
    return report


def _fmt(pct: float | None) -> str:
    return f"{pct:.1f}%" if pct is not None else "n/a"


def _print_report(report: dict) -> None:
    print("=" * 60)
    print("RAG Evaluation Report")
    print("=" * 60)
    for i, d in enumerate(report["details"], start=1):
        print(f"\n[{i}] {d['question']}")
        print(f"    retrieved sources: {d['retrieved_sources']}")
        print(f"    answer: {d['answer']}")
        if d["answerable"]:
            print(f"    expected source: {d['expected_source']}")
            print(f"    retrieval hit (top-{report['top_k']}): {'YES' if d['retrieval_hit'] else 'NO'}")
            print(f"    retrieval hit (top-1, strict): {'YES' if d['retrieval_hit_top1'] else 'NO'}")
            print(
                f"    keyword hit: {'YES' if d['answer_hit'] else 'NO'} "
                f"(matched {d['matched_keywords']} of {d['expected_keywords']})"
            )
        else:
            print("    expected: a refusal (nothing in the corpus answers this)")
            print(f"    correctly refused: {'YES' if d['refused'] else 'NO'}")
    print("\n" + "=" * 60)
    print(f"Total questions:               {report['total_questions']}")
    print(f"Retrieval accuracy (top-{report['top_k']}):    {_fmt(report['retrieval_accuracy'])}")
    print(f"Retrieval accuracy (top-1):    {_fmt(report['retrieval_accuracy_top1'])}")
    print(f"Answer keyword-hit rate:       {_fmt(report['answer_keyword_hit_rate'])}")
    print(f"Refusal accuracy (unanswerable): {_fmt(report['refusal_accuracy'])}")
    print("=" * 60)
