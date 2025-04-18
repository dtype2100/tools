from fastapi import FastAPI
import os, shutil
from schemas.model_request import ModelRequest
from backend.utils.path_handler import AiModelPath
from backend.crud.ai_download import AiModelDownload
from backend.services.ai_load import AiLoad

app = FastAPI()
loaded_models = {}

@app.post("/download")
def download_model(req: ModelRequest):
    apath = AiModelPath()
    model_path = apath.get_model_path(req.model_name, req.save_path)
    try:
        amcrud = AiModelDownload(req.model_name, model_path, req.model_format)
        return amcrud.model_download()
    except Exception as e:
        return {"error": str(e)}


@app.post("/load")
def load_model(req: ModelRequest):
    try:
        apath = AiModelPath()
        model_path = apath.get_model_path(req.model_name, req.save_path)
        amcrud = AiLoad(req.model_name, model_path, req.model_format)
        return amcrud.model_load(loaded_models)
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
    apath = AiModelPath()
    model_path = apath.get_model_path(req.model_name, req.save_path)
    if os.path.exists(model_path):
        shutil.rmtree(model_path, ignore_errors=True)
    loaded_models.pop(req.model_name, None)
    return {"message": f"{req.model_name} deleted."}

@app.get("/list_models")
def list_models():
    apath = AiModelPath()
    model_dir = apath.DEFAULT_MODEL_DIR
    if not os.path.exists(model_dir):
        return {"models": []}
    models = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ]
    return {"models": models}
