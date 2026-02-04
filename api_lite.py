from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import onnxruntime as ort
import numpy as np
import cv2
import io
from PIL import Image
import os
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best.onnx"
session = None
input_name = None
output_names = None

# YOLOv8 Constants
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

@app.on_event("startup")
async def startup_event():
    global session, input_name, output_names
    if os.path.exists(MODEL_PATH):
        print(f"Loading ONNX model from {MODEL_PATH}...")
        # Load ONNX model
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        print("Model loaded successfully (Lite Mode).")
    else:
        print(f"WARNING: {MODEL_PATH} not found.")

def preprocess_image(image: Image.Image):
    # Resize and pad to maintain aspect ratio
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
    
    h, w, _ = img.shape
    scale = min(IMG_SIZE / h, IMG_SIZE / w)
    nh, nw = int(h * scale), int(w * scale)
    
    img_resized = cv2.resize(img, (nw, nh))
    
    # Create canvas
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = img_resized
    
    # Normalize and reshape
    blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)
    return blob, scale

def postprocess(outputs, scale, original_size):
    # YOLOv8 output: (1, 4 + class_count, 8400)
    predictions = np.squeeze(outputs[0]).T # Shape: (8400, 4 + class_count)
    
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > CONF_THRESHOLD, :]
    scores = scores[scores > CONF_THRESHOLD]
    
    if len(scores) == 0:
        return []

    class_ids = np.argmax(predictions[:, 4:], axis=1)
    
    # Extract boxes
    boxes = predictions[:, :4]
    
    # XYWH to XYXY
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    
    boxes_xyxy = np.stack((x1, y1, x2, y2), axis=1)
    
    # Rescale back to original image
    boxes_xyxy /= scale
    
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes_xyxy[i]
            results.append({
                "box": box.tolist(),
                "confidence": float(scores[i]),
                "class": int(class_ids[i]),
                "name": f"class_{int(class_ids[i])}" # Map this if you have class names
            })
            
    return results

@app.get("/")
def index():
    return {"status": "Online", "mode": "Ultra Lite (ONNXRuntime)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_size = image.size
    
    # Preprocess
    input_tensor, scale = preprocess_image(image)
    
    # Inference
    outputs = session.run(output_names, {input_name: input_tensor})
    
    # Postprocess
    detections = postprocess(outputs, scale, original_size)
    
    # Draw boxes
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    best_detection = None
    max_conf = -1
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        conf = det["confidence"]
        
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det['name']} {conf:.2f}"
        cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if conf > max_conf:
            max_conf = conf
            best_detection = det

    # Convert back to base64
    _, buffer = cv2.imencode('.jpg', img_cv)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "filename": file.filename,
        "mode": "Ultra Lite",
        "count": len(detections),
        "best_detection": best_detection,
        "detections": detections,
        "image_base64": img_base64
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
