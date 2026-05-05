# setup7_2-qwen Code Submission

This directory is a self-contained TIRA code submission for the
`advertisement-in-retrieval-augmented-generation-2026` task. The container
entrypoint is `/predict.py`. At runtime it loads the bundled Longformer
sequence-classification model from `/models/setup7_2-qwen`, consumes the input
`qwen` neutral reference directly, and writes `predictions.jsonl`.

This is a classifier-only submission. It does not run Qwen chat generation at
inference time.

## Submission Package Contents

The package contains these files:

- `predict.py`: runtime inference entrypoint used by TIRA
- `Dockerfile`: image definition used by `tira-cli code-submission`
- `requirements.txt`: Python dependencies installed into the container
- `.dockerignore`: excludes local caches and outputs from the image context
- `tools/docker`: optional Docker wrapper for `tira-cli` compatibility
- `README.md`: submission specification and operator notes

The image preloads the published Hugging Face model
`sambus211/zhaw_at_touche_setup7_2_qwen` during Docker build and saves it as a
local bundle in `/models/setup7_2-qwen`, so the final TIRA runtime stays
offline.

## Runtime Contract

TIRA will execute the submission with:

```bash
/predict.py
```

The runner supports both direct CLI usage and the standard TIRA environment
variables:

- `inputDataset`: dynamic input directory mounted by TIRA
- `outputDir`: dynamic output directory mounted by TIRA

Equivalent CLI flags are also supported:

- `--dataset`: TIRA dataset id, local directory, or local JSONL file
- `--input-directory`: explicit local or mounted input directory
- `--output-directory`: explicit output directory
- `--output`: explicit output file path

If the input is a directory, `predict.py` automatically discovers the most
likely JSONL file whose rows contain `id`, `query`, `response`, and `qwen`.

## Input Specification

Each input row must be a JSON object with at least these fields:

- `id`: unique row identifier
- `query`: user query string
- `response`: generated answer to classify
- `qwen`: neutral Qwen reference for the same query

Example row:

```json
{"id":"example-1","query":"What is the best way to compare CRM pricing?","response":"HubSpot offers flexible pricing tiers and powerful automation tools for growing teams.","qwen":"Comparing CRM pricing usually involves checking subscription tiers, user limits, included features, support options, and any additional implementation or usage costs."}
```

The model input text is built exactly like this:

```text
USER QUERY: <query>

NEUTRAL REFERENCE (QWEN): <qwen>

RESPONSE TO CLASSIFY: <response>

LABEL THIS AS AD OR NEUTRAL:
```

If `qwen` is missing or empty, the submission fails fast. It does not generate
fallback neutrals.

## Output Specification

The submission writes one file:

```text
predictions.jsonl
```

Default location under TIRA:

```text
$outputDir/predictions.jsonl
```

Each output row is a JSON object like:

```json
{"id":"example-1","label":1,"ad_prob":0.9973}
```

Field semantics:

- `id`: copied from the input row
- `label`: binary integer prediction in `{0, 1}`
- `ad_prob`: `softmax(logits)[1]`

The output row order follows the input row order.

## Model And Inference Defaults

- Hugging Face source: `sambus211/zhaw_at_touche_setup7_2_qwen`
- Bundled local model dir: `/models/setup7_2-qwen`
- Architecture: Longformer sequence classifier
- Input format: `query_neutral_response`
- Reference field: `qwen`
- Reference label: `QWEN`
- Default batch size: `4`
- Default max length: `1024`
- Padding mode: `padding="max_length"`
- Default threshold: `0.5`
- Default device selection: `cuda`, then `mps`, then `cpu`

Override values if needed:

```bash
./predict.py \
  --dataset ../zhaw_at_touche/data/generated/qwen/responses-validation-with-neutral_qwen.jsonl \
  --output ./out/predictions.jsonl \
  --model-dir /absolute/path/to/setup7_2-qwen \
  --batch-size 4 \
  --max-length 1024 \
  --threshold 0.5 \
  --device cpu
```

## Local Verification

For direct host-side runs, make sure you already have a local exported model
bundle and point `--model-dir` at it. If you do not, prefer the Docker dry-run
validation below because the container build creates `/models/setup7_2-qwen`
for you.

Run on a local Qwen-enriched JSONL file:

```bash
./predict.py \
  --dataset ../zhaw_at_touche/data/generated/qwen/responses-validation-with-neutral_qwen.jsonl \
  --model-dir /absolute/path/to/setup7_2-qwen \
  --output ./out/predictions.jsonl
```

Or run against a TIRA dataset id through the TIRA Python client:

```bash
./predict.py \
  --dataset advertisement-in-retrieval-augmented-generation-2026/ads-in-rag-task-1-detection-spot-check-20260422-training \
  --model-dir /absolute/path/to/setup7_2-qwen \
  --output ./out/predictions.jsonl
```

The TIRA-style environment variables also work directly:

```bash
inputDataset=../zhaw_at_touche/data/generated/qwen outputDir=./out ./predict.py \
  --model-dir /absolute/path/to/setup7_2-qwen
```

If you point the submission at a raw task directory that does not already
contain a `qwen` field, it will fail by design.

## Validate The Docker Submission

Use this section before uploading to TIRA to validate that the Dockerized
submission behaves like a real TIRA run.

### Prerequisites

- Docker is installed and running
- `tira` is installed: `pip3 install tira`
- you are registered for the task in TIRA
- for real uploads, the git repository is clean: `git status`

Authenticate and verify the local TIRA client:

```bash
tira-cli login --token <YOUR_TIRA_TOKEN>
tira-cli verify-installation --task advertisement-in-retrieval-augmented-generation-2026
```

If you use Docker Desktop with the containerd image store enabled, TIRA may
reject uploaded images even though the local build and push succeed. In that
case, force Docker v2 manifest output during submission:

```bash
tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py' \
  --build-args '--output type=docker --provenance=false'
```

If Docker still exports an incompatible image, disable Docker Desktop's
`Use containerd for pulling and storing images` setting, rebuild, and retry the
submission.

If the failure happens before your submission image is built, `tira-cli` may be
rejecting its own internal `tira-mini` preflight image before the build args
above are applied. In that case, prepend the repo-local Docker wrapper so every
`docker build` invoked by `tira-cli` gets the compatibility flags:

```bash
PATH="${PWD}/tools:${PATH}" tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

### TIRA Dry-Run Validation

This is the closest local validation to a real TIRA code submission. It builds
the Docker image from this directory and runs the submission on the specified
dataset without uploading anything.

```bash
tira-cli code-submission \
  --dry-run \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

What this validates:

- the Docker image builds successfully
- `/predict.py` starts correctly inside the container
- the runtime can read `$inputDataset`
- the runtime writes a valid JSONL prediction file to `$outputDir`
- the output format is acceptable for the task

### Optional Local Sandbox Test With `tira-run`

Build the image:

```bash
docker build -t zhaw-at-touche-setup7-2-qwen-local .
```

Run the image with TIRA-style directory mounts:

```bash
mkdir -p ./tira-output
tira-run \
  --input-directory ../zhaw_at_touche/data/generated/qwen \
  --output-directory ./tira-output \
  --image zhaw-at-touche-setup7-2-qwen-local \
  --command /predict.py
```

The resulting file will be:

```text
./tira-output/predictions.jsonl
```
