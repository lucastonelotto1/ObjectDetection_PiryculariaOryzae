from ultralytics import YOLO

# Load the PyTorch model
model = YOLO("best.pt")

# Export the model to ONNX format
# imgsz=640 ensures the input size is fixed, which helps with optimization
print("Exporting model to ONNX...")
model.export(format="onnx", imgsz=640)
print("Export complete! 'best.onnx' should be in your folder.")
