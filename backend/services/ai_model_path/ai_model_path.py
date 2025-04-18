import os

class  AiModelPath:
    def __init__(self, default_model_dir: str = "./ai_models"):
        self.DEFAULT_MODEL_DIR = default_model_dir

    def get_model_path(self, model_name, save_path):
        return save_path if save_path else os.path.join(self.DEFAULT_MODEL_DIR, model_name)