# 🩺 Pneumonia Detection with Explainable AI (Grad-CAM + FastAPI)

This project detects Pneumonia from chest X-ray images using a DenseNet model and provides visual explanations using Grad-CAM.  
It includes a web interface built with FastAPI and auto-generated diagnostic reports in PDF.

## 🚀 Features
- Upload chest X-rays (JPG, PNG, DICOM)
- AI prediction (Normal vs Pneumonia)
- Grad-CAM visualization
- Adjustable contrast & transparency
- PDF report generator
- Ready for cloud deployment (AWS/GCP)

## 🧠 Model
- Architecture: DenseNet121 (transfer learning)
- Dataset: [ChestX-ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## 🖥️ Run Locally
```bash
git clone https://github.com/<your-username>/pneumonia-detection.git
cd pneumonia-detection
pip install -r requirements.txt
uvicorn src.predict_api:app --reload
