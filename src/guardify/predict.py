"""
Prediction CLI for Guardify bundles.
"""

from __future__ import annotations

import argparse
import json

from guardify.inference import InferenceService


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with a Guardify model bundle.")
    parser.add_argument("--bundle", required=True, help="Path to a model bundle.")
    parser.add_argument("--text", required=True, help="Text to classify.")
    args = parser.parse_args()
    service = InferenceService(args.bundle)
    result = service.predict_text(args.text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

