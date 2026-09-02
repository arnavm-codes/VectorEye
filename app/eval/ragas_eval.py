"""Retrieval quality evaluation using RAGAS's ID-based context metrics.

Scoped deliberately to what's applicable for a retrieval-only pipeline: this
system never generates a textual answer grounded in video content (see the
vault note's "Why not a video-LLM?" section), so RAGAS's generation-facing
metrics (faithfulness, answer relevancy/correctness) don't apply here -- there
is no "answer" to check for groundedness. What *does* apply is retrieval
quality: given a query, did Qdrant return the right clips, ranked well?

IDBasedContextPrecision / IDBasedContextRecall score that directly by
comparing retrieved clip IDs against a hand-labeled reference set -- pure ID
matching, no LLM judge involved. This keeps evaluation consistent with the
project's "no LLM understands the video" constraint: ground truth here is a
human watching the footage and labeling relevant clips, not an LLM guessing.

Usage: uv run python -m app.eval.ragas_eval
Requires: data/eval_queries.json filled in with real queries + ground truth,
and the pipeline already indexed (scripts/run_pipeline.py).
"""

import json
import warnings
from pathlib import Path

from app.api.search import search_clips
from app.config import PROJECT_ROOT

EVAL_QUERIES_PATH = PROJECT_ROOT / "data" / "eval_queries.json"


def load_eval_queries() -> list[dict]:
    with open(EVAL_QUERIES_PATH) as f:
        data = json.load(f)
    return data.get("queries", [])


def run_eval(top_k: int = 5) -> list[dict]:
    warnings.filterwarnings("ignore", category=DeprecationWarning, message="Importing.*ragas.metrics.*deprecated")
    from ragas import SingleTurnSample
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    queries = load_eval_queries()
    if not queries:
        print(
            f"No eval queries defined in {EVAL_QUERIES_PATH}. "
            "Add {'query': ..., 'relevant_clips': [...]} entries once footage is indexed."
        )
        return []

    precision_metric = IDBasedContextPrecision()
    recall_metric = IDBasedContextRecall()

    rows = []
    for item in queries:
        results = search_clips(item["query"], top_k=top_k)
        retrieved_ids = [Path(r["clip_path"]).name for r in results]

        sample = SingleTurnSample(
            user_input=item["query"],
            retrieved_context_ids=retrieved_ids,
            reference_context_ids=item["relevant_clips"],
        )
        precision = precision_metric.single_turn_score(sample)
        recall = recall_metric.single_turn_score(sample)

        rows.append(
            {
                "query": item["query"],
                "retrieved": retrieved_ids,
                "relevant": item["relevant_clips"],
                "context_precision": precision,
                "context_recall": recall,
            }
        )

    return rows


def print_report(rows: list[dict]):
    if not rows:
        return

    for row in rows:
        print(f"\nQuery: {row['query']}")
        print(f"  Retrieved:  {row['retrieved']}")
        print(f"  Relevant:   {row['relevant']}")
        print(f"  Context precision: {row['context_precision']:.3f}")
        print(f"  Context recall:    {row['context_recall']:.3f}")

    mean_precision = sum(r["context_precision"] for r in rows) / len(rows)
    mean_recall = sum(r["context_recall"] for r in rows) / len(rows)
    print(f"\n=== Mean over {len(rows)} queries ===")
    print(f"Context precision: {mean_precision:.3f}")
    print(f"Context recall:    {mean_recall:.3f}")


if __name__ == "__main__":
    print_report(run_eval())
