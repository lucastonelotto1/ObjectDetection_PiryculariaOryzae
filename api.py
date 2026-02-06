from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import os
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PT_MODEL_PATH = "best.pt"

model = None

@app.on_event("startup")
async def startup_event():
    global model
    if os.path.exists(PT_MODEL_PATH):
        print(f"Loading PT model from {PT_MODEL_PATH}...")
        print("WARNING: Using PT model may cause OOM on free tier servers.")
        model = YOLO(PT_MODEL_PATH)
        # Explicitly set class names to ensure correct mapping
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            model.model.names[0] = 'no_pyr'
            model.model.names[1] = 'pyr'
        else:
            # Fallback if structure is different (e.g. older versions/generic wrapper)
            model.names[0] = 'no_pyr'
            model.names[1] = 'pyr'

        print(f"Model names set to: {model.names}")
        print("PT Model loaded successfully.")
    else:
        print(f"WARNING: No model found in {os.getcwd()}")

@app.get("/")
def index():
    return {"status": "Online", "model": "YOLOv8", "documentation": "/docs"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
    
    # Validate file type (if provided)
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Inference
        results = model.predict(source=image, imgsz=640, conf=0.25, save=False)
        
        detections = []
        best_detection = None
        max_conf = -1.0

        annotated_image = None

        # Prepare image for drawing (PIL -> OpenCV BGR)
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                bbox = box.xyxy[0].tolist() 
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw boxes with thicker black lines
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 0), 4)
                
                label = f"{name} {conf:.2f}"
                # Larger black text
                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

                det = {
                    "class": name,
                    "confidence": round(conf, 4),
                    "box": bbox
                }
                detections.append(det)

                # Track best detection
                if conf > max_conf:
                    max_conf = conf
                    best_detection = det
        
        # Convert back to PIL (BGR -> RGB)
        annotated_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Encode image to base64
        import base64
        buffered = io.BytesIO()
        if annotated_image:
            annotated_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            img_str = None

        return {
            "filename": file.filename, 
            "count": len(detections),
            "best_detection": best_detection,
            "detections": detections,
            "image_base64": img_str
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
