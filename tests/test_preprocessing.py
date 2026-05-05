from __future__ import annotations

import pandas as pd

from guardify.preprocessing import PreprocessingPipeline, preprocess_dataframe


def test_preprocessing_handles_noisy_text():
    pipeline = PreprocessingPipeline()
    result = pipeline.normalize("B***h 😂 you are stuuuupid!!! #clown @user https://a.b")

    assert "[laugh]" in result.normalized_text
    assert "[ABUSIVE]" in result.normalized_text
    assert "stupid" in result.flagged_tokens
    assert "clown" in result.flagged_tokens


def test_dataframe_preprocessing_adds_columns():
    df = pd.DataFrame([{"text": "Tum bohot cheap ho"}, {"text": "All good"}])
    processed = preprocess_dataframe(df)
    assert "normalized_text" in processed.columns
    assert "flagged_tokens" in processed.columns
