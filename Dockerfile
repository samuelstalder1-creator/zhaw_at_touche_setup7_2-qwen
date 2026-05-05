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

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

model_repo = "${MODEL_REPO}"
model_dir = Path("${MODEL_DIR}")
qwen_model_repo = "${QWEN_MODEL_REPO}"
qwen_model_dir = Path("${QWEN_MODEL_DIR}")
model_dir.mkdir(parents=True, exist_ok=True)
qwen_model_dir.mkdir(parents=True, exist_ok=True)

classifier_tokenizer = AutoTokenizer.from_pretrained(model_repo)
classifier_model = AutoModelForSequenceClassification.from_pretrained(model_repo)
classifier_tokenizer.save_pretrained(model_dir)
classifier_model.save_pretrained(model_dir)

qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_repo)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_repo)
qwen_tokenizer.save_pretrained(qwen_model_dir)
qwen_model.save_pretrained(qwen_model_dir)
PY

RUN rm -rf /root/.cache/huggingface

ENTRYPOINT ["python3", "/predict.py"]
