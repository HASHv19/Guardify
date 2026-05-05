from __future__ import annotations

import types

from guardify.models.transformer import create_transformer_model


class DummyModel:
    config = types.SimpleNamespace(hidden_size=8)

    def __call__(self, *args, **kwargs):  # pragma: no cover
        return None


class DummyTokenizer:
    pass


def test_transformer_factory(monkeypatch):
    monkeypatch.setattr(
        "guardify.models.transformer.AutoModel.from_pretrained",
        lambda *args, **kwargs: DummyModel(),
    )
    monkeypatch.setattr(
        "guardify.models.transformer.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    assets = create_transformer_model(model_name="dummy/muril", num_labels=2, dropout=0.2)

    assert assets.model.model_name == "dummy/muril"
    assert isinstance(assets.tokenizer, DummyTokenizer)

