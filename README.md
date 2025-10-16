# Pneumonia Detection using Deep Learning 🩺

A FastAPI-based web service for detecting pneumonia from chest X-ray DICOM images using a DenseNet model.

## Features
- Upload DICOM or jpeg images for analysis
- Outputs a downloadable PDF report with patient info and prediction
- REST API built with FastAPI

## How to Run
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/pneumonia-detection.git
cd pneumonia-detection

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # (Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn src.predict_api:app --reload --port 8000
