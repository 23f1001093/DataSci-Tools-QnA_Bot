from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np

app = FastAPI()

@app.post("/captcha")
async def solve_captcha(file: UploadFile = File(...)):
    # Read image from uploaded file
    image_data = await file.read()

    # Convert image bytes to PIL Image, then to NumPy array
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_np = np.array(image)

    # Convert RGB to Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Apply binary thresholding (invert colors if needed)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # OCR configuration
    config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(thresh, config=config)

    # Clean up the OCR output: Keep only alphanumeric and symbols if needed
    clean_text = raw_text.strip()

    return JSONResponse(
        content={
            "answer": clean_text,
            "email": "23f1001093@ds.study.iitm.ac.in"
        },
        status_code=200
    )
