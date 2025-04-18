from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download
from llama_cpp import Llama
import os, shutil
from models.model_request import ModelRequest
from services.ai_model_path import AiModelPath

app = FastAPI()
apath = AiModelPath()
loaded_models = {}

@app.post("/download")
def download_model(req: ModelRequest):
    model_path = apath.get_model_path(req.model_name, req.save_path)

    if req.model_format == "gguf":
        if not os.path.exists(model_path):
            return {"error": f"GGUF 파일이 {model_path} 에 존재하지 않습니다. 수동으로 복사하세요."}
        return {"message": f"{req.model_name} (GGUF) is ready at {model_path}"}

    if os.path.exists(model_path):
        return {"message": f"{req.model_name} already exists at {model_path}"}

    try:
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(req.model_name, local_dir=model_path)
        return {"message": f"{req.model_name} downloaded to {model_path}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/load")
def load_model(req: ModelRequest):
    model_path = apath.get_model_path(req.model_name, req.save_path)

    if req.model_name in loaded_models:
        return {"message": f"{req.model_name} already loaded."}

    if req.model_format == "gguf":
        if not os.path.exists(model_path):
            return {"error": f"GGUF 파일이 {model_path} 에 없습니다."}
        try:
            llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)
            loaded_models[req.model_name] = {"type": "gguf", "pipe": llm}
            return {"message": f"{req.model_name} (GGUF) loaded from {model_path}"}
        except Exception as e:
            return {"error": str(e)}

    if not os.path.exists(model_path):
        return {"error": f"{model_path} 경로에 transformers 모델이 없습니다. 먼저 다운로드하세요."}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        loaded_models[req.model_name] = {"type": "transformers", "pipe": pipe}
        return {"message": f"{req.model_name} loaded from {model_path}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/infer")
def infer(req: ModelRequest, prompt: str):
    entry = loaded_models.get(req.model_name)
    if not entry:
        return {"error": f"{req.model_name} not loaded. 먼저 로드해주세요."}

    try:
        if entry["type"] == "gguf":
            result = entry["pipe"](prompt)
            return {"result": result["choices"][0]["text"]}
        else:
            result = entry["pipe"](prompt, max_new_tokens=100)
            return {"result": result[0]['generated_text']}
    except Exception as e:
        return {"error": str(e)}

@app.post("/delete")
def delete_model(req: ModelRequest):
    model_path = apath.get_model_path(req.model_name, req.save_path)
    if os.path.exists(model_path):
        shutil.rmtree(model_path, ignore_errors=True)
    loaded_models.pop(req.model_name, None)
    return {"message": f"{req.model_name} deleted."}

@app.get("/list_models")
def list_models():
    model_dir = apath.DEFAULT_MODEL_DIR
    if not os.path.exists(model_dir):
        return {"models": []}
    models = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ]
    return {"models": models}
