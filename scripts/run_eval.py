"""CLI entry point: python -m scripts.run_eval"""

from src.evaluate import evaluate

if __name__ == "__main__":
    report = evaluate()
    print(report)
