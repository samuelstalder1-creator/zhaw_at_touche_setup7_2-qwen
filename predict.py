#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = Path("/models/setup7_2-qwen")
DEFAULT_QWEN_MODEL_DIR = Path("/models/qwen2.5-1.5b-instruct")
DEFAULT_QWEN_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_QWEN_MODEL_NAME = str(DEFAULT_QWEN_MODEL_DIR)
REFERENCE_FIELD = "qwen"
REFERENCE_LABEL = "QWEN"
DEFAULT_TAG = "zhawAtToucheSetup72Qwen"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 1024
DEFAULT_CPU_BATCH_SIZE = 1
DEFAULT_CPU_MAX_LENGTH = 512
DEFAULT_MAX_NEW_TOKENS = 220
DEFAULT_THRESHOLD = 0.5
UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
LIST_PREFIX_RE = re.compile(r"^\s*(?:[•◦▪●‣∙]|[-*+]|(?:\d+|[a-zA-Z])[.)])\s+")

SYSTEM_PROMPT = """Goal:
Write a helpful, factual answer to the user's query that matches the style of existing neutral responses.

Rules:
- Do not mention brand names, companies, vendors, product models, or specific services.
- Do not promote or recommend a specific item.
- Avoid marketing language, persuasion, links, or calls to action.
- Generic product or technical terms are allowed.

Style Requirements:
- Write in flowing prose using natural sentences and short paragraphs.
- Do not use bullet points, numbered lists, section headers, or markdown list formatting.
- Keep tone factual, balanced, and conversational.
- Return exactly one continuous paragraph.
- Do not output any newline characters.

Length:
- Target roughly 130-200 words unless the query is trivial.
"""


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


def clean_response_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = UNICODE_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 16)), cleaned)
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\t", " ")

    paragraphs: list[str] = []
    current_lines: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            if current_lines:
                paragraphs.append(" ".join(current_lines).strip())
                current_lines = []
            continue
        line = LIST_PREFIX_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            current_lines.append(line)

    if current_lines:
        paragraphs.append(" ".join(current_lines).strip())

    return re.sub(r"\s+", " ", " ".join(paragraphs).strip())


def build_chat_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query.strip()},
    ]


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


def has_non_empty_text_field(row: dict[str, Any], field_name: str) -> bool:
    value = row.get(field_name)
    return isinstance(value, str) and bool(value.strip())


def input_candidate_score(
    path: Path,
    row: dict[str, Any],
    *,
    prefer_reference_field: str | None = None,
) -> tuple[int, int, str] | None:
    if "id" not in row or "query" not in row or "response" not in row:
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

    # For reference-aware setups, prefer already enriched files over raw task
    # files so runtime matches the local validation condition whenever possible.
    if prefer_reference_field and has_non_empty_text_field(row, prefer_reference_field):
        score += 1000
        if prefer_reference_field in name:
            score += 20
        if "neutral" in name:
            score += 10

    return score, -len(path.parts), str(path)


def discover_input_file(input_path: Path, *, prefer_reference_field: str | None = None) -> Path:
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
        score = input_candidate_score(path, row, prefer_reference_field=prefer_reference_field)
        if score is not None:
            candidates.append((score, path))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a response JSONL file under {input_path}. "
            "Expected rows with at least 'id', 'query', and 'response' fields."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def validate_record(row: dict[str, Any], *, origin: str) -> dict[str, Any]:
    row_id = row.get("id")
    query = row.get("query")
    response = row.get("response")

    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError(f"{origin} is missing a valid 'id'.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{origin} is missing a valid 'query'.")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"{origin} is missing a valid 'response'.")

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


def load_records_from_source(
    input_source: str,
    *,
    prefer_reference_field: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    input_path = Path(input_source)
    if input_path.exists():
        input_file = discover_input_file(input_path, prefer_reference_field=prefer_reference_field)
        return load_records(input_file), str(input_file)

    return load_tira_dataset_records(input_source), input_source


def load_local_generation_model(model_name: str, device: str):
    model_kwargs: dict[str, Any] = {}
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            **model_kwargs,
        ).to(device)
    except (OSError, ValueError):
        fallback_model_name = DEFAULT_QWEN_MODEL_REPO if model_name == DEFAULT_QWEN_MODEL_NAME else model_name
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(fallback_model_name, **model_kwargs).to(device)

    model.eval()
    return tokenizer, model


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


def generate_neutral_response(
    *,
    tokenizer,
    model,
    query: str,
    device: str,
    max_new_tokens: int,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support chat templates for local Qwen generation.")

    prompt_text = tokenizer.apply_chat_template(
        build_chat_messages(query),
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in tokenized.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=model.dtype)
        if device == "cuda" and isinstance(model.dtype, torch.dtype)
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast_ctx:
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    generated_ids = generated[:, input_length:]
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if not text:
        raise RuntimeError("Empty neutral response from local Qwen generation.")
    return clean_response_text(text)


def maybe_generate_neutrals(
    *,
    records: list[dict[str, Any]],
    qwen_tokenizer,
    qwen_model,
    qwen_device: str,
    max_new_tokens: int,
    reuse_existing_neutral: bool,
) -> tuple[list[dict[str, Any]], int]:
    enriched_records: list[dict[str, Any]] = []
    query_cache: dict[str, str] = {}
    generated_queries = 0

    for record in records:
        out = dict(record)
        existing = out.get(REFERENCE_FIELD)
        if reuse_existing_neutral and isinstance(existing, str) and existing.strip():
            enriched_records.append(out)
            continue

        query = out.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Record {out.get('id', '<unknown>')} is missing a valid 'query' field.")

        neutral = query_cache.get(query)
        if neutral is None:
            neutral = generate_neutral_response(
                tokenizer=qwen_tokenizer,
                model=qwen_model,
                query=query,
                device=qwen_device,
                max_new_tokens=max_new_tokens,
            )
            query_cache[query] = neutral
            generated_queries += 1

        out[REFERENCE_FIELD] = neutral
        enriched_records.append(out)

    return enriched_records, generated_queries


def needs_neutral_generation(records: list[dict[str, Any]], *, reuse_existing_neutral: bool) -> bool:
    if not reuse_existing_neutral:
        return True
    return any(
        not isinstance(record.get(REFERENCE_FIELD), str)
        or not str(record.get(REFERENCE_FIELD, "")).strip()
        for record in records
    )


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
            for record, text in zip(batch, texts):
                print(f"[predict] id={record.get('id')} input:\n{text}\n---")
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
            for record, probability in zip(batch, probabilities):
                label = 1 if probability >= threshold else 0
                print(f"[predict] id={record.get('id')} ad_prob={probability:.4f} label={label}")
                predictions.append(Prediction(label=label, ad_prob=float(probability)))
    return predictions


def write_predictions(
    *,
    records: list[dict[str, Any]],
    predictions: list[Prediction],
    output_file: Path,
    tag: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record, prediction in zip(records, predictions):
            row = {
                "id": record["id"],
                "label": prediction.label,
                "tag": tag,
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


def tune_runtime_settings(
    *,
    batch_size: int,
    max_length: int,
    device: str,
    user_batch_size: int | None,
    user_max_length: int | None,
) -> tuple[int, int]:
    if device == "cuda":
        return batch_size, max_length

    if user_batch_size is None:
        batch_size = DEFAULT_CPU_BATCH_SIZE

    if user_max_length is None:
        max_length = min(max_length, DEFAULT_CPU_MAX_LENGTH)

    return batch_size, max_length


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the setup7_2-qwen TIRA submission: generate or reuse Qwen neutrals, then classify with the bundled Longformer model."
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
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_MODEL_NAME,
        help="Local or remote Qwen generator model identifier.",
    )
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--max-length", type=int, default=None, help="Max token length override.")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--reuse-existing-neutral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing 'qwen' field in the input if present; otherwise generate it locally.",
    )
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_source = resolve_input_source(args)
    preferred_reference_field = REFERENCE_FIELD if args.reuse_existing_neutral else None
    raw_records, input_description = load_records_from_source(
        input_source,
        prefer_reference_field=preferred_reference_field,
    )
    output_file = resolve_output_file(args)

    device = resolve_device(args.device)
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
    max_length = args.max_length if args.max_length is not None else DEFAULT_MAX_LENGTH
    batch_size, max_length = tune_runtime_settings(
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        user_batch_size=args.batch_size,
        user_max_length=args.max_length,
    )

    records = raw_records
    generated_queries = 0
    if needs_neutral_generation(raw_records, reuse_existing_neutral=args.reuse_existing_neutral):
        qwen_tokenizer, qwen_model = load_local_generation_model(args.qwen_model, device)
        records, generated_queries = maybe_generate_neutrals(
            records=raw_records,
            qwen_tokenizer=qwen_tokenizer,
            qwen_model=qwen_model,
            qwen_device=device,
            max_new_tokens=args.qwen_max_new_tokens,
            reuse_existing_neutral=args.reuse_existing_neutral,
        )
        del qwen_model
        del qwen_tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    tokenizer, model = load_model(Path(args.model_dir), device)
    predictions = predict_records(
        records=records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        threshold=args.threshold,
    )
    write_predictions(records=records, predictions=predictions, output_file=output_file, tag=args.tag)

    print(f"input_source={input_description}")
    print(f"rows={len(records)}")
    print(f"output_file={output_file}")
    print(f"model_dir={args.model_dir}")
    print(f"qwen_model={args.qwen_model}")
    print(f"tag={args.tag}")
    print(f"generated_qwen_neutrals={generated_queries}")


if __name__ == "__main__":
    main()
