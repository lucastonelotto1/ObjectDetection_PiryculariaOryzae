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
    Rotates a point back to the original image coordinates.
    """
    x, y = point
    cx_rot, cy_rot = center_rot
    cx_orig, cy_orig = center_orig
    
    # Convert angle to radians
    rad = np.radians(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    # Translate to center of rotated image
    tx = x - cx_rot
    ty = y - cy_rot
    
    # Rotate
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
    
    # Rotate corners
    rotated_corners = [rotate_point(c, -angle, center_rot, center_orig) for c in corners]
    
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
    
    # Prepare image for drawing (PIL -> NumPy -> OpenCV BGR)
    img_np = np.array(image)
    
    # Check if image is grayscale (2 dimensions) or has 3 dimensions
    if len(img_np.shape) == 2:
        # Grayscale to BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        # RGBA to BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    elif img_np.shape[2] == 3:
        # RGB to BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    img_cv = img_np.copy()

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

            if conf > max_conf:
                max_conf = conf
                best_detection = det
    
    # Convert back to PIL (BGR -> RGB)
    annotated_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    # Encode image to base64
    import base64
    buffered = io.BytesIO()
    annotated_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "count": len(detections),
        "best_detection": best_detection,
        "max_conf": max_conf,
        "detections": detections,
        "image_base64": img_str,
        "annotated_image_obj": annotated_image # Keep object if needed for further processing
    }

async def predict_with_tta(model, image: Image.Image, filename: str):
    """
    Performs Test Time Augmentation (TTA) by rotating the image and selecting the best prediction.
    """
    best_overall_response = None
    highest_overall_conf = -1.0
    
    # Default response (usually the 0-degree or first processed)
    default_response = None

    # Capture original image size for later cropping
    orig_w, orig_h = image.size

    for angle in ANGLES:
        # Rotate image
        rot_img = rotate_image(image, angle)
        
        # Predict
        results = model.predict(source=rot_img, imgsz=640, conf=0.25, save=False)
        
        # Process results
        processed = process_detection_results(results, rot_img, model)
        
        response_payload = {
            "filename": filename,
            "rotation_angle": angle,
            "count": processed["count"],
            "best_detection": processed["best_detection"],
            "detections": processed["detections"],
            "image_base64": processed["image_base64"],
            "annotated_image_obj": processed["annotated_image_obj"]
        }

        # Set default if it's the first angle (0 degrees usually)
        if default_response is None:
            default_response = response_payload

        # Check if this rotation yielded a better result
        if processed["max_conf"] > highest_overall_conf:
            highest_overall_conf = processed["max_conf"]
            best_overall_response = response_payload
            
    
    # If we found a best response, we need to rotate the annotated image back to original orientation
    if best_overall_response:
        best_angle = best_overall_response["rotation_angle"]
        annotated_img = best_overall_response["annotated_image_obj"]
        
        # Inverse rotation: rotate by -angle
        # We use the same background color (black) or grey as requested? 
        # User accepted black in previous turn diff. 
        # Keep consistent with rotate_image function which now uses black (0,0,0) based on user manual edit.
        final_img = annotated_img.rotate(-best_angle, expand=True, fillcolor=(0, 0, 0))
        
        # Center Crop to remove the massive black borders from expansion
        # The image is now upright (conceptually) but on a large canvas.
        # We want to crop the center to the original size.
        curr_w, curr_h = final_img.size
        left = (curr_w - orig_w) / 2
        top = (curr_h - orig_h) / 2
        right = (curr_w + orig_w) / 2
        bottom = (curr_h + orig_h) / 2
        
        final_img = final_img.crop((left, top, right, bottom))
        
        # Update base64
        import base64
        buffered = io.BytesIO()
        final_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        best_overall_response["image_base64"] = img_str
        
        # Remove the PIL object before returning (not JSON serializable)
        if "annotated_image_obj" in best_overall_response:
            del best_overall_response["annotated_image_obj"]

        # Rotate bounding boxes back to original coordinates
        if best_angle != 0:
             # We need the size of the ROTATED image before it was cropped (which is what detections are based on)
             # But wait, `annotated_img` IS the rotated image (with detections drawn on it, but we have the raw detection data)
             # Actually, `annotated_image_obj` in the response is the one with boxes drawn. 
             # The detections in `best_overall_response["detections"]` are based on `rot_img` size.
             # We need to recalculate `rot_img` size or store it. 
             # However, since `annotated_image_obj` corresponds to `rot_img`, we can use its size.
             # BUT `annotated_image_obj` has already been deleted from the dict a few lines up? No, we just deleted the key from dict, the var `annotated_img` still holds the object?
             # Actually line 176 `del best_overall_response["annotated_image_obj"]` removes it from the dict.
             # We assigned `annotated_img = best_overall_response["annotated_image_obj"]` on line 147. So `annotated_img` is safe.
             
             rot_w, rot_h = annotated_img.size
             
             # Update best_detection box
             if best_overall_response.get("best_detection"):
                 old_box = best_overall_response["best_detection"]["box"]
                 new_box = rotate_bounding_box(old_box, best_angle, (orig_w, orig_h), (rot_w, rot_h))
                 best_overall_response["best_detection"]["box"] = new_box
            
             # Update all detections
             for det in best_overall_response.get("detections", []):
                 old_box = det["box"]
                 new_box = rotate_bounding_box(old_box, best_angle, (orig_w, orig_h), (rot_w, rot_h))
                 det["box"] = new_box

    # Also clean default_response if it exists and we're returning it or if it shares reference
    if default_response and "annotated_image_obj" in default_response:
        del default_response["annotated_image_obj"]

    return best_overall_response if best_overall_response and highest_overall_conf > -1.0 else default_response

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
