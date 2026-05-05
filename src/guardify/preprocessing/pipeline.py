"""
Text normalization pipeline for bilingual social text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .lexicon import ABUSE_LEXICON, EMOJI_MAP, OBFUSCATION_PATTERNS


def _optional_transliterate(text: str, enabled: bool = False) -> str:
    if not enabled:
        return text
    try:
        from indic_transliteration import sanscript  # type: ignore
        from indic_transliteration.sanscript import transliterate  # type: ignore

        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    except Exception:
        return text


@dataclass
class PreprocessResult:
    original_text: str
    normalized_text: str
    flagged_tokens: list[str]


class PreprocessingPipeline:
    """
    Applies deterministic normalization and abuse token extraction.
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        transliterate: bool = False,
        keep_hashtag_text: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.transliterate = transliterate
        self.keep_hashtag_text = keep_hashtag_text

    def _replace_emojis(self, text: str) -> str:
        for emoji_char, replacement in EMOJI_MAP.items():
            text = text.replace(emoji_char, replacement)
        return text

    def _clean_obfuscation(self, text: str) -> str:
        normalized = text
        for pattern, replacement in OBFUSCATION_PATTERNS.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        return normalized

    def _normalize_hashtags(self, text: str) -> str:
        if self.keep_hashtag_text:
            return re.sub(r"#(\w+)", r"\1", text)
        return re.sub(r"#\w+", " ", text)

    def _flag_tokens(self, text: str) -> tuple[str, list[str]]:
        flagged: list[str] = []
        normalized = text
        for token, replacement in ABUSE_LEXICON.items():
            pattern = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", flags=re.IGNORECASE)
            if pattern.search(normalized):
                flagged.append(token)
                normalized = pattern.sub(replacement, normalized)
        return normalized, sorted(set(flagged))

    def normalize(self, text: Any) -> PreprocessResult:
        if pd.isna(text) or not str(text).strip():
            return PreprocessResult(str(text) if not pd.isna(text) else "", "", [])

        original = str(text)
        normalized = original
        normalized = self._replace_emojis(normalized)
        normalized = _optional_transliterate(normalized, enabled=self.transliterate)
        normalized = re.sub(r"http\S+|www\S+|https\S+", " ", normalized, flags=re.MULTILINE)
        normalized = re.sub(r"@\w+", " ", normalized)
        normalized = self._normalize_hashtags(normalized)
        normalized = self._clean_obfuscation(normalized)
        normalized = re.sub(r"(.)\1{2,}", r"\1", normalized)
        normalized = re.sub(r"\b[_\-\.\*]+", "", normalized)
        normalized = re.sub(r"[^\w\s\u0600-\u06FF\u0900-\u097F\[\],.!?]", " ", normalized)
        if self.lowercase:
            normalized = normalized.lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized, flagged = self._flag_tokens(normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return PreprocessResult(original, normalized, flagged)


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    text_column: str = "text",
    pipeline: PreprocessingPipeline | None = None,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Missing text column '{text_column}'.")
    pipeline = pipeline or PreprocessingPipeline()
    processed = df.copy()
    results = processed[text_column].apply(pipeline.normalize)
    processed["normalized_text"] = results.apply(lambda item: item.normalized_text)
    processed["flagged_tokens"] = results.apply(lambda item: item.flagged_tokens)
    return processed

