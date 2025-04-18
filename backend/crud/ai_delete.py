import os, shutil

class AiModelDelete:
    def __init__(self, model_path) -> None:
       self.model_path = model_path 
        
    def delete_model(self):
        model_path = self.model_path
        
        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
            