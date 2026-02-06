# Object Detection API - Pirycularia Oryzae

This API provides an object detection service using a YOLOv8 model to identify "pyr" (Pirycularia Oryzae) and "no_pyr" classes in images.

## Endpoint: `/predict`

**Method**: `POST`

**Description**: Upload one or multiple images to detect objects. The API processes all uploaded images and returns the detection results for the single image that contains the highest confidence detection.

### Request Format

The API expects a `multipart/form-data` request.

- **Key**: `file` (This key is used for all images).
- **Value**: Image file(s) (JPEG, PNG, etc.).

You can append multiple files with the same key `file`.

### Frontend Example (JavaScript/Fetch)

```javascript
const formData = new FormData();
const fileInput = document.querySelector('input[type="file"]');

// Append all selected files to the same 'file' key
for (let i = 0; i < fileInput.files.length; i++) {
    formData.append("file", fileInput.files[i]);
}

try {
    const response = await fetch("https://your-api-url.com/predict", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Prediction Result:", data);
    
    // Display the annotated image
    if (data.image_base64) {
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${data.image_base64}`;
        document.body.appendChild(img);
    }
    
} catch (error) {
    console.error("Error:", error);
}
```

### Response Format (JSON)

The response contains details about the best detection found among all uploaded images.

```json
{
    "filename": "image_name.jpg",           // Name of the file with the best detection
    "count": 2,                             // Number of objects detected in that image
    "best_detection": {                     // Details of the single highest confidence detection
        "class": "pyr",
        "confidence": 0.9521,
        "box": [100.5, 200.0, 300.5, 400.0] // [x1, y1, x2, y2] coordinates
    },
    "detections": [                         // List of all detections in that image
        {
            "class": "pyr",
            "confidence": 0.9521,
            "box": [100.5, 200.0, 300.5, 400.0]
        },
        {
            "class": "no_pyr",
            "confidence": 0.8500,
            "box": [50.0, 60.0, 150.0, 200.0]
        }
    ],
    "image_base64": "..."                   // Base64 encoded string of the annotated image (with bounding boxes)
}
```

- **`filename`**: The name of the image file that produced these results.
- **`count`**: Total number of bounding boxes detected in the "winner" image.
- **`best_detection`**: Object containing the class, confidence, and bounding box of the single highest confidence score found.
- **`detections`**: Array of all detection objects found in the image.
- **`image_base64`**: The processed image with black bounding boxes and labels drawn on it, encoded in base64 (ready to be displayed in an `<img>` tag).
