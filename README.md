# setup7_2-qwen Code Submission

This submission now runs the full Qwen-plus-Longformer inference pipeline for
`advertisement-in-retrieval-augmented-generation-2026`:

1. load the bundled `Qwen/Qwen2.5-1.5B-Instruct` generator
2. generate a neutral response from the input `query` when `qwen` is missing
3. build the Longformer prompt from `query + qwen + response`
4. classify with the bundled `setup7_2-qwen` Longformer model

The runtime entrypoint is `/predict.py`. It writes `predictions.jsonl` in TIRA
format.

## Package Contents

- `predict.py`: runtime inference entrypoint
- `Dockerfile`: container build for TIRA
- `requirements.txt`: Python dependencies
- `tools/docker`: optional Docker wrapper for local `tira-cli` workflows
- `README.md`: operator notes

During Docker build the image stores two local model bundles so runtime stays
offline:

- classifier: `/models/setup7_2-qwen`
- generator: `/models/qwen2.5-1.5b-instruct`

## Runtime Contract

TIRA executes:

```bash
/predict.py
```

Supported inputs:

- `inputDataset`: TIRA input directory
- `outputDir`: TIRA output directory
- `--dataset`: TIRA dataset id, local directory, or JSONL file
- `--input-directory`: explicit local input directory
- `--output-directory`: explicit output directory
- `--output`: explicit prediction file
- `--tag`: optional submission tag written to each prediction row

If the input is a directory, `predict.py` discovers the most likely JSONL file
whose rows contain at least `id`, `query`, and `response`. When
`--reuse-existing-neutral` is enabled and the directory contains both raw task
files and Qwen-enriched JSONL files, it prefers the file that already contains
the `qwen` field so the runtime reuses the pre-generated neutral responses
instead of regenerating them.

## Input Specification

Each row must contain:

- `id`
- `query`
- `response`

Optional:

- `qwen`: precomputed neutral response for the same query

Example raw input row:

```json
{"id":"example-1","query":"What is the best way to compare CRM pricing?","response":"HubSpot offers flexible pricing tiers and powerful automation tools for growing teams."}
```

If `qwen` is missing or empty, the submission generates it with the bundled
Qwen model. If `qwen` is already present, it is reused by default. When a
directory contains both raw and enriched files, the enriched file is preferred
for the same reason. Disable reuse with `--no-reuse-existing-neutral`.

The classifier input is built exactly as:

```text
USER QUERY: <query>

NEUTRAL REFERENCE (QWEN): <generated_or_existing_qwen>

RESPONSE TO CLASSIFY: <response>

LABEL THIS AS AD OR NEUTRAL:
```

## Output Specification

The submission writes:

```text
$outputDir/predictions.jsonl
```

Each row is:

```json
{"id":"example-1","label":1,"tag":"zhawAtToucheSetup72Qwen"}
```

- `id`: copied from input
- `label`: binary prediction in `{0, 1}`
- `tag`: submission identifier, defaulting to `zhawAtToucheSetup72Qwen`

## Defaults

- classifier dir: `/models/setup7_2-qwen`
- generator dir: `/models/qwen2.5-1.5b-instruct`
- generator source model: `Qwen/Qwen2.5-1.5B-Instruct`
- classifier max length: `1024`
- classifier batch size: `4`
- generator max new tokens: `220`
- output tag: `zhawAtToucheSetup72Qwen`
- threshold: `0.5`
- device order: `cuda`, then `mps`, then `cpu`

CLI override example:

```bash
./predict.py \
  --dataset ../zhaw_at_touche/data/task/responses-validation.jsonl \
  --model-dir /absolute/path/to/setup7_2-qwen \
  --qwen-model Qwen/Qwen2.5-1.5B-Instruct \
  --output ./out/predictions.jsonl \
  --device cpu
```

## Local Verification

Run directly on raw task data:

```bash
./predict.py \
  --dataset ../zhaw_at_touche/data/task/responses-validation.jsonl \
  --model-dir /absolute/path/to/setup7_2-qwen \
  --qwen-model Qwen/Qwen2.5-1.5B-Instruct \
  --output ./out/predictions.jsonl
```

Run on already enriched data and reuse the existing `qwen` field:

```bash
./predict.py \
  --dataset ../zhaw_at_touche/data/generated/qwen/responses-validation-with-neutral_qwen.jsonl \
  --model-dir /absolute/path/to/setup7_2-qwen \
  --qwen-model Qwen/Qwen2.5-1.5B-Instruct \
  --output ./out/predictions.jsonl
```

TIRA-style environment variables also work:

```bash
inputDataset=../zhaw_at_touche/data/task outputDir=./out ./predict.py \
  --model-dir /absolute/path/to/setup7_2-qwen \
  --qwen-model Qwen/Qwen2.5-1.5B-Instruct
```

## Docker Validation

Build locally:

```bash
docker build -t zhaw-at-touche-setup7-2-qwen-local .
```

Dry-run the submission with `tira-cli`:

```bash
tira-cli code-submission \
  --dry-run \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

Or run the built image with `tira-run`:

```bash
mkdir -p ./tira-output
tira-run \
  --input-directory ../zhaw_at_touche/data/task \
  --output-directory ./tira-output \
  --image zhaw-at-touche-setup7-2-qwen-local \
  --command /predict.py
```
