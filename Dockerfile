FROM webis/touche25-ad-detection:0.0.1

ADD predict.py /predict.py
ADD requirements.txt /requirements.txt

RUN pip3 install --no-cache-dir -r /requirements.txt

ARG MODEL_REPO=sambus211/zhaw_at_touche_setup7_2_qwen
ARG MODEL_DIR=/models/setup7_2-qwen

RUN python3 - <<PY
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_repo = "${MODEL_REPO}"
model_dir = Path("${MODEL_DIR}")
model_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForSequenceClassification.from_pretrained(model_repo)

tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)
PY

RUN rm -rf /root/.cache/huggingface

ENTRYPOINT ["python3", "/predict.py"]
