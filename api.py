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
# Prioritize ONNX model for performance/memory on Render
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
async def predict(files: list[UploadFile] = File(..., alias="file")):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    best_response = None
    highest_confidence = -1.0
    
    # If no detections are found in any image, we'll want to return *something*.
    # We can default to the result of the first video/image processed.
    default_response = None

    for file in files:
        # Validate file type (if provided)
        if file.content_type and not file.content_type.startswith("image/"):
            # We can skip non-images or raise error. 
            # For now, let's skip/continue or error. 
            # Given the user context, raising error might be safer to avoid confusion.
             raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image.")

        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Inference
            results = model.predict(source=image, imgsz=640, conf=0.25, save=False)
            
            detections = []
            best_detection_in_image = None
            max_conf_in_image = -1.0
            
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

                    # Track best detection inside this image
                    if conf > max_conf_in_image:
                        max_conf_in_image = conf
                        best_detection_in_image = det
            
            # Encode image to base64
            import base64
            buffered = io.BytesIO()
            if annotated_image:
                annotated_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                img_str = None

            response_payload = {
                "filename": file.filename, 
                "count": len(detections),
                "best_detection": best_detection_in_image,
                "detections": detections,
                "image_base64": img_str
            }

            # If this is the first image processed, set it as default fallback
            if default_response is None:
                default_response = response_payload
            
            # Check if this image has a better detection than what we've seen so far
            if max_conf_in_image > highest_confidence:
                highest_confidence = max_conf_in_image
                best_response = response_payload

        except Exception as e:
            # If one image fails, ideally we might want to continue or report partial error.
            # But simpler to fail fast or just log and continue. 
            # Let's log and continue to next image to allow partial success?
            # Or fail strictly? The query asks for "predict", imply strict.
            # However, traceback printing is in original code.
            import traceback
            traceback.print_exc()
            # We will raise to be safe, or we could just skip this file.
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")

    # Return the best response found, or the default (first image) if no detections in any
    return best_response if best_response else default_response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
