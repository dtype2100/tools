import os
from huggingface_hub import snapshot_download

class AiModelDownload:
    def __init__(self, model_name: str, model_path: str, model_format: str) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.model_format = model_format

    def model_download(self):
        """
        Input parameters below \n
        model_path: str \n
        model_name: str \n
        model_format: str \n
        """
        model_path = self.model_path
        model_name = self.model_name
        model_format = self.model_format

        if model_format == "gguf":
            if not os.path.exists(model_path):
                return {"error": f"GGUF 파일이 {model_path} 에 존재하지 않습니다. 수동으로 복사하세요."}
            return {"message": f"{model_name} (GGUF) is ready at {model_path}"}

        if os.path.exists(model_path):
            return {"message": f"{model_name} already exists at {model_path}"}

        os.makedirs(model_path, exist_ok=True)
        snapshot_download(model_name, local_dir=model_path)
        
        return {"message": f"{model_name} downloaded to {model_path}"}