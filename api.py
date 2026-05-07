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
ANGLES = [0, 22.5, 45, 67.5, 90, -22.5, -45, -67.5, -90]

model = None

def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """
    Rotates the image by the given angle.
    Expands the image to prevent cropping and fills the background with gray (114, 114, 114).
    """
    return image.rotate(angle, expand=True, fillcolor=(0, 0, 0))

def rotate_point(point, angle, center_rot, center_orig):
    """
    Maps a point from the rotated image back to the original image coordinates.
    PIL rotates counter-clockwise for positive angles, so the inverse rotation
    uses the NEGATED angle.
    """
    x, y = point
    cx_rot, cy_rot = center_rot
    cx_orig, cy_orig = center_orig
    
    # Inverse rotation: negate the angle to undo what PIL did
    rad = np.radians(-angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    # Translate to center of rotated image
    tx = x - cx_rot
    ty = y - cy_rot
    
    # Apply inverse rotation
    rx = tx * cos_a - ty * sin_a
    ry = tx * sin_a + ty * cos_a
    
    # Translate back to center of original image
    final_x = rx + cx_orig
    final_y = ry + cy_orig
    
    return final_x, final_y

def rotate_bounding_box(bbox, angle, orig_size, rot_size):
    """
    Rotates a bounding box back to the original image coordinates.
    """
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = orig_size
    rot_w, rot_h = rot_size
    
    center_rot = (rot_w / 2, rot_h / 2)
    center_orig = (orig_w / 2, orig_h / 2)
    
    # Corners of the bounding box
    corners = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2)
    ]
    
    # Rotate corners (map from rotated image back to original coordinates)
    # PIL rotates counter-clockwise for positive angles; using the same sign here
    # keeps the mapping consistent with how `rotate_image` was applied.
    rotated_corners = [rotate_point(c, angle, center_rot, center_orig) for c in corners]
    
    # Find new bounding box (AABB)
    xs = [c[0] for c in rotated_corners]
    ys = [c[1] for c in rotated_corners]
    
    min_x = max(0, min(xs))
    max_x = min(orig_w, max(xs))
    min_y = max(0, min(ys))
    max_y = min(orig_h, max(ys))
    
    return [min_x, min_y, max_x, max_y]

def process_detection_results(results, image: Image.Image, model) -> dict:
    """
    Processes YOLO results and returns a structured dictionary with detections and the annotated image.
    """
    detections = []
    max_conf = -1.0
    best_detection = None

    for r in results:
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

            if conf > max_conf:
                max_conf = conf
                best_detection = det

    return {
        "count": len(detections),
        "best_detection": best_detection,
        "max_conf": max_conf,
        "detections": detections
    }

async def predict_with_tta(model, image: Image.Image, filename: str):
    """
    Performs TTA by trying all rotation angles. Picks the rotation with the
    highest confidence, draws boxes on that rotated image, then rotates the
    entire annotated image back to the original orientation — boxes stay
    perfectly aligned without any coordinate math.
    """
    import base64

    # Save original dimensions to crop back after inverse rotation
    orig_w, orig_h = image.size

    best_conf = -1.0
    best_processed = None
    best_rot_img = None
    best_angle = 0

    for angle in ANGLES:
        rot_img = rotate_image(image, angle)
        results = model.predict(source=rot_img, imgsz=640, conf=0.25, save=False)
        processed = process_detection_results(results, rot_img, model)

        if processed["max_conf"] > best_conf:
            best_conf = processed["max_conf"]
            best_processed = processed
            best_rot_img = rot_img
            best_angle = angle

    # Draw boxes directly on the winning rotated image
    base_img = best_rot_img.convert("RGB")
    img_np = np.array(base_img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for det in best_processed["detections"]:
        x1, y1, x2, y2 = map(int, det["box"])
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Rotate the whole annotated image back to original orientation.
    # Since boxes are already painted as pixels, they rotate with the image
    # and remain perfectly aligned — no coordinate transformation needed.
    annotated_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    if best_angle != 0:
        annotated_img = annotated_img.rotate(-best_angle, expand=True, fillcolor=(0, 0, 0))
        # Crop center back to original dimensions to remove black borders
        w, h = annotated_img.size
        left = (w - orig_w) // 2
        top  = (h - orig_h) // 2
        annotated_img = annotated_img.crop((left, top, left + orig_w, top + orig_h))

    # Encode annotated image to base64
    final_img = annotated_img
    buffered = io.BytesIO()
    final_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "filename": filename,
        "rotation_angle": best_angle,
        "count": best_processed["count"],
        "best_detection": best_processed["best_detection"],
        "detections": best_processed["detections"],
        "image_base64": img_str,
    }

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
             raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image.")

        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Apply EXIF orientation so phone/camera images aren't double-rotated
            try:
                from PIL import ImageOps
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass  # If EXIF data is missing or malformed, skip
            
            # Perform TTA prediction
            best_img_response = await predict_with_tta(model, image, file.filename)
            
            # Check if this image (from the batch of uploaded files) has the best detection so far
            current_max_conf = -1.0
            if best_img_response and best_img_response.get("best_detection"):
                 current_max_conf = best_img_response["best_detection"]["confidence"]
            
            if default_response is None:
                default_response = best_img_response
                
            if current_max_conf > highest_confidence:
                highest_confidence = current_max_conf
                best_response = best_img_response

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")

    # Return the best response found, or the default (first image result) if no detections in any
    return best_response if best_response else default_response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
