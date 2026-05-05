#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = Path("/models/setup7_2-qwen")
REFERENCE_FIELD = "qwen"
REFERENCE_LABEL = "QWEN"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 1024
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Prediction:
    label: int
    ad_prob: float


def resolve_device(requested: str | None) -> str:
    if requested is not None:
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda' is not available.")
        if requested == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise ValueError("Requested device 'mps' is not available.")
        return requested

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def autocast_context(device: str):
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_model_input(record: dict[str, Any]) -> str:
    query = record.get("query", "")
    response = record.get("response", "")
    neutral = record.get(REFERENCE_FIELD, "")

    if not isinstance(query, str):
        query = ""
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Record {record.get('id', '<unknown>')} is missing a valid 'response' field.")
    if not isinstance(neutral, str) or not neutral.strip():
        raise ValueError(f"Record {record.get('id', '<unknown>')} is missing a valid '{REFERENCE_FIELD}' field.")

    return (
        f"USER QUERY: {query}\n\n"
        f"NEUTRAL REFERENCE ({REFERENCE_LABEL}): {neutral}\n\n"
        f"RESPONSE TO CLASSIFY: {response}\n\n"
        "LABEL THIS AS AD OR NEUTRAL:"
    )


def first_jsonl_row(path: Path) -> dict[str, Any] | None:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row in {path} is not a JSON object.")
            return row
    return None


def input_candidate_score(path: Path, row: dict[str, Any]) -> tuple[int, int, str] | None:
    if "id" not in row or "query" not in row or "response" not in row or REFERENCE_FIELD not in row:
        return None

    name = path.name.lower()
    if not name.endswith(".jsonl"):
        return None
    if "label" in name:
        return None

    if name == "responses.jsonl":
        score = 100
    elif name == "responses-test.jsonl":
        score = 95
    elif name == "responses-validation.jsonl":
        score = 90
    elif name == "responses-train.jsonl":
        score = 85
    elif name.startswith("responses-"):
        score = 80
    elif "responses" in name:
        score = 70
    elif "response" in name:
        score = 60
    else:
        score = 50

    return score, -len(path.parts), str(path)


def discover_input_file(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path must be a directory or JSONL file: {input_path}")

    candidates: list[tuple[tuple[int, int, str], Path]] = []
    for path in sorted(input_path.rglob("*.jsonl")):
        row = first_jsonl_row(path)
        if row is None:
            continue
        score = input_candidate_score(path, row)
        if score is not None:
            candidates.append((score, path))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a Qwen-enriched response JSONL file under {input_path}. "
            f"Expected rows with at least 'id', 'query', 'response', and '{REFERENCE_FIELD}' fields."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def validate_record(row: dict[str, Any], *, origin: str) -> dict[str, Any]:
    row_id = row.get("id")
    query = row.get("query")
    response = row.get("response")
    neutral = row.get(REFERENCE_FIELD)

    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError(f"{origin} is missing a valid 'id'.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{origin} is missing a valid 'query'.")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"{origin} is missing a valid 'response'.")
    if not isinstance(neutral, str) or not neutral.strip():
        raise ValueError(f"{origin} is missing a valid '{REFERENCE_FIELD}'.")

    return row


def load_records(input_file: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {input_file} is not a JSON object.")
            records.append(validate_record(row, origin=f"Line {line_number} in {input_file}"))
    if not records:
        raise ValueError(f"Input file is empty: {input_file}")
    return records


def load_tira_dataset_records(dataset: str) -> list[dict[str, Any]]:
    records = Client().pd.inputs(dataset).to_dict(orient="records")
    if not records:
        raise ValueError(f"TIRA dataset resolved to zero input rows: {dataset}")

    normalized_records: list[dict[str, Any]] = []
    for index, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"TIRA input row {index} is not a JSON object.")
        normalized_records.append(validate_record(row, origin=f"TIRA input row {index}"))
    return normalized_records


def load_records_from_source(input_source: str) -> tuple[list[dict[str, Any]], str]:
    input_path = Path(input_source)
    if input_path.exists():
        input_file = discover_input_file(input_path)
        return load_records(input_file), str(input_file)

    return load_tira_dataset_records(input_source), input_source


def load_model(model_dir: Path, device: str):
    if not model_dir.exists():
        raise FileNotFoundError(f"Bundled model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def predict_records(
    *,
    records: list[dict[str, Any]],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_length: int,
    threshold: float,
) -> list[Prediction]:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    predictions: list[Prediction] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [build_model_input(record) for record in batch]
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in tokenized.items()}
            with autocast_context(device):
                logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            for probability in probabilities:
                label = 1 if probability >= threshold else 0
                predictions.append(Prediction(label=label, ad_prob=float(probability)))
    return predictions


def write_predictions(
    *,
    records: list[dict[str, Any]],
    predictions: list[Prediction],
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record, prediction in zip(records, predictions):
            row = {
                "id": record["id"],
                "label": prediction.label,
                "ad_prob": prediction.ad_prob,
            }
            handle.write(json.dumps(row) + "\n")


def resolve_input_source(args: argparse.Namespace) -> str:
    if args.dataset and args.input_directory and args.dataset != args.input_directory:
        raise ValueError("Pass only one of --dataset or --input-directory.")
    input_source = args.input_directory or args.dataset
    if not input_source:
        raise ValueError("Pass --dataset/--input-directory or set the inputDataset environment variable.")
    return input_source


def resolve_output_file(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    if args.output_directory:
        return Path(args.output_directory) / "predictions.jsonl"
    return Path(get_output_directory(str(Path(__file__).parent))) / "predictions.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the setup7_2-qwen Longformer TIRA submission on a TIRA dataset id or local input directory."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="TIRA dataset id, local input directory, or local JSONL file.",
    )
    parser.add_argument(
        "--input-directory",
        default=os.environ.get("inputDataset"),
        help="Dynamic TIRA input directory. Defaults to $inputDataset.",
    )
    parser.add_argument(
        "--output-directory",
        default=os.environ.get("outputDir"),
        help="Dynamic TIRA output directory. Defaults to $outputDir.",
    )
    parser.add_argument(
        "--output",
        "--output-file",
        dest="output",
        default=None,
        help="Optional explicit prediction file path. Overrides --output-directory when set.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Local bundled Hugging Face sequence-classification model directory.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_source = resolve_input_source(args)
    records, input_description = load_records_from_source(input_source)
    output_file = resolve_output_file(args)

    device = resolve_device(args.device)
    tokenizer, model = load_model(Path(args.model_dir), device)
    predictions = predict_records(
        records=records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threshold=args.threshold,
    )
    write_predictions(records=records, predictions=predictions, output_file=output_file)

    print(f"input_source={input_description}")
    print(f"rows={len(records)}")
    print(f"output_file={output_file}")
    print(f"model_dir={args.model_dir}")


if __name__ == "__main__":
    main()
