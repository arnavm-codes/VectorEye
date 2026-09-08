"""Lightweight (attribute, object) extraction from a text query, for the
attribute-binding fix (see vault note "Attribute-binding / bag-of-words
retrieval failure", 2026-09-08). Uses spaCy's dependency parse (an
adjectival-modifier edge from attribute to its head noun) rather than a
hardcoded word list or an LLM call, so it generalizes to any object/
attribute pair the query names -- not just colors, not just vehicles.
"""

import spacy

_nlp = None


def _load_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_attribute_object_pairs(query: str) -> list[tuple[str, str]]:
    """Returns (attribute, object) pairs found via adjectival-modifier
    dependencies, e.g. "find clips with blue cars" -> [("blue", "car")].
    Empty list if no adjective directly modifies a noun in the query --
    callers should fall back to plain CLIP search in that case.
    """
    nlp = _load_nlp()
    doc = nlp(query)
    pairs = []
    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ in ("NOUN", "PROPN"):
            pairs.append((token.text.lower(), token.head.lemma_.lower()))
    return pairs
