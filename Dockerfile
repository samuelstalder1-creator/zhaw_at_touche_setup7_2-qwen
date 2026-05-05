FROM webis/touche25-ad-detection:0.0.1

ADD predict.py /predict.py
ADD requirements.txt /requirements.txt

RUN pip3 install --no-cache-dir -r /requirements.txt
RUN pip3 uninstall -y torchvision

ARG MODEL_REPO=sambus211/zhaw_at_touche_setup7_2_qwen
ARG MODEL_DIR=/models/setup7_2-qwen
ARG QWEN_MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct
ARG QWEN_MODEL_DIR=/models/qwen2.5-1.5b-instruct

RUN python3 - <<PY
from pathlib import Path

from huggingface_hub import snapshot_download


def download_snapshot(repo_id: str, target_dir: str) -> None:
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(path),
    )


download_snapshot("${MODEL_REPO}", "${MODEL_DIR}")
download_snapshot("${QWEN_MODEL_REPO}", "${QWEN_MODEL_DIR}")
PY

RUN rm -rf /root/.cache/huggingface

ENTRYPOINT ["python3", "/predict.py"]
