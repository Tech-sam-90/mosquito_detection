# main.py
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import os

# Initialize the FastAPI app
app = FastAPI(title="Mosquito Counter API")

# ---
# IMPORTANT: Load your trained model
# Before deploying, make sure your 'best.pt' file is in the same directory
# as this script, or provide the correct path.
# ---
# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')

# Check if the model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please make sure 'best.pt' is in the same directory.")

# Load your custom-trained YOLOv8 model
model = YOLO(model_path)

@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "Mosquito Counter API is running."}


@app.post("/count-mosquitoes/")
async def count_mosquitoes(file: UploadFile = File(...)):
    """
    This endpoint receives an image file, runs inference using the YOLOv8 model,
    and returns the number of detected mosquitoes.
    """
    # 1. Read the image file from the request
    image_bytes = await file.read()
    
    # 2. Convert the bytes into a PIL Image
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Invalid image file provided. Details: {e}"}

    # 3. Run inference on the image
    # We only care about the 'mosquito' class. Let's assume its class ID is 0.
    # You can adjust the confidence threshold as needed.
    results = model(image, conf=0.4) 

    # 4. Count the number of detected mosquitoes
    mosquito_count = 0
    
    # The result object contains bounding boxes for all detected objects.
    # We need to iterate through them and count only the ones classified as 'mosquito'.
    # Note: The class ID for 'mosquito' depends on your 'data.yaml' file.
    # It is usually 0 if it's the first (or only) class.
    # Replace 0 with the correct class ID if necessary.
    
    # Assuming 'mosquito' is class ID 2 from your training log
    mosquito_class_id = 2 
    
    for result in results:
        # Each result object has a 'boxes' attribute
        for box in result.boxes:
            # Each box has a 'cls' attribute for its class ID
            if int(box.cls) == mosquito_class_id:
                mosquito_count += 1

    # 5. Return the count in a JSON response
    return {"mosquito_count": mosquito_count}