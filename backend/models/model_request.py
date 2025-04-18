from pydantic import BaseModel

class ModelRequest(BaseModel):
    model_name: str
    save_path: str = None
    model_format: str = "transformers"  # "transformers" or "gguf"