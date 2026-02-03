from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import os
import cv2

app = FastAPI()

# Load model globally to avoid reloading on every request
# Assuming best.pt is in the same directory
MODEL_PATH = "best.pt"

model = None

@app.on_event("startup")
async def startup_event():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"WARNING: {MODEL_PATH} not found in {os.getcwd()}")

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

        for r in results:
            # Generate annotated image (BGR -> RGB -> PIL -> Base64)
            im_bgr = r.plot()
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(im_rgb)

            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                bbox = box.xyxy[0].tolist() 
                
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
