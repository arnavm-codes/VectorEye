"""Groq chat layer: query cleanup + result explanation only.

Deliberately isolated from retrieval -- never sees or reasons about video
content, only the user's text query and the already-retrieved clip
metadata. If Groq is unavailable, callers can skip this and use raw
search_clips() results directly.
"""

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def clean_query(raw_query: str) -> str:
    """Rewrites a conversational user query into a concise search phrase."""
    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's request as a short, concrete visual "
                    "search phrase describing what to look for in a video "
                    "clip (objects, actions, scene). Output only the phrase, "
                    "no explanation."
                ),
            },
            {"role": "user", "content": raw_query},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def explain_results(raw_query: str, results: list[dict]) -> str:
    """Turns raw search hits into a short conversational summary."""
    if not results:
        return "No matching clips were found for that query."

    lines = [
        f"- camera {r['camera_id']}, {r['start_ts']}-{r['end_ts']}s, score {r['score']:.3f}"
        for r in results
    ]
    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You present video clip search results conversationally. "
                    "You did NOT watch the clips -- these are nearest-neighbor "
                    "embedding matches, not confirmed content. Never assert or "
                    "imply what a clip actually shows (e.g. do not say 'this "
                    "clip shows X'). Only report the similarity scores and "
                    "metadata (camera, time range) given, and tell the user "
                    "these are candidate matches to review themselves. Be brief."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {raw_query}\nResults:\n" + "\n".join(lines),
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
