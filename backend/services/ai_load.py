import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_cpp import  Llama

class AiLoad:
    def __init__(self, model_name, model_path, model_format) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.model_format = model_format

    def model_load(self, loaded_models: dict):
        """
        모델을 로드하고 loaded_models 딕셔너리에 저장합니다.
        """
        model_name = self.model_name
        model_path = self.model_path
        model_format = self.model_format

        if model_name in loaded_models:
            return {"message": f"{model_name} already loaded."}

        if model_format == "gguf":
            if not os.path.exists(model_path):
                return {"error": f"GGUF 파일이 {model_path} 에 없습니다."}
            try:
                llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)
                loaded_models[model_name] = {"type": "gguf", "pipe": llm}
                return {"message": f"{model_name} (GGUF) loaded from {model_path}"}
            except Exception as e:
                return {"error": str(e)}

        if not os.path.exists(model_path):
            return {"error": f"{model_path} 경로에 transformers 모델이 없습니다. 먼저 다운로드하세요."}
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        loaded_models[model_name] = {"type": "transformers", "pipe": pipe}
        return {"message": f"{model_name} loaded from {model_path}"}
    