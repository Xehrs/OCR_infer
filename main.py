from fastapi import FastAPI,File, UploadFile
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import asyncio
from inference import single_task_recognition, model_setup

class OCRModelManager:
    def __init__(self):
        self.models = {}
        self.is_ready = False
    
    async def load_models(self):
        """Load all required OCR models"""
        try:
            # Load your models here
            # TODO: Make config path a parameter
            self.models['ocr'] = model_setup(config_path="MonkeyOCR/model_configs.yaml")
            
            self.is_ready = True
            print("All models loaded successfully")
        except Exception as e:
            print(f"Failed to load models: {e}")
            raise
    
    def get_model(self, model_type: str):
        if not self.is_ready:
            raise HTTPException(status_code=503, detail="Models not ready")
        return self.models.get(model_type)

# Global model manager
model_manager = OCRModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting OCR service...")
    await model_manager.load_models()
    print("OCR service ready!")
    
    yield
    
    # Shutdown
    print("Shutting down OCR service...")

app = FastAPI(lifespan=lifespan)

class ParseRequest(BaseModel):
    file_path: str
    task: str

class ParseResponse(BaseModel):
    success: bool
    message: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "OCR API is running"}

@app.post("/parse", response_model=ParseResponse)
async def parse_document(request: ParseRequest):
    """Parse complete document (PDF or image)"""

    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    ocr_model = model_manager.get_model('ocr')
    return ParseResponse(success=True, message=single_task_recognition(request.file_path, ocr_model, request.task))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)