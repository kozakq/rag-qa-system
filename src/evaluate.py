"""
Retrieval + answer quality evaluation against eval/questions.json.

This is the module that proves the system actually works, not just that it runs.
A resume line like "Model Evaluation" needs a real report behind it — this produces one.

TODO (implementation):
- Load eval/questions.json: [{"question": str, "expected_source": str, "expected_keywords": [str]}]
- For each question: retrieve top_k chunks, check whether expected_source appears among them
  (retrieval accuracy), then generate an answer and check whether expected_keywords appear in
  it (rough answer-quality proxy — good enough for a portfolio project, no need for anything
  more elaborate)
- Print/save a small report: retrieval accuracy %, answer-keyword-hit %, and per-question detail
- Should be runnable as: python -m scripts.run_eval
"""


def evaluate() -> dict:
    raise NotImplementedError
