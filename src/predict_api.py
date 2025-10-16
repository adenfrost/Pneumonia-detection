from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np
import io
import tempfile
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tensorflow.keras.models import load_model
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import imghdr

app = FastAPI()

MODEL_PATH = "saved_models/densenet_pneumonia.h5"
model = load_model(MODEL_PATH)

# ----------------------------------------------------
# Utility function: read DICOM safely
# ----------------------------------------------------
def read_dicom_as_image(path):
    ds = pydicom.dcmread(path, force=True)
    # Handle missing Transfer Syntax UID
    if "TransferSyntaxUID" not in ds.file_meta:
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    try:
        data = apply_voi_lut(ds.pixel_array, ds)
    except Exception:
        data = ds.pixel_array

    # Normalize and convert to RGB
    data = data.astype(float)
    data -= np.min(data)
    data /= np.max(data)
    data = np.stack([data]*3, axis=-1)  # grayscale -> RGB
    img = Image.fromarray((data * 255).astype(np.uint8)).resize((224, 224))
    img = np.array(img) / 255.0
    return img, ds

# ----------------------------------------------------
# Utility function: make PDF report
# ----------------------------------------------------
def make_pdf_report(patient_info, predicted_prob, label):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 60, "Pneumonia Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(40, height - 100, f"Patient Name: {patient_info.get('PatientName', 'Unknown')}")
    c.drawString(40, height - 120, f"Patient ID: {patient_info.get('PatientID', 'Unknown')}")
    c.drawString(40, height - 140, f"Study Date: {patient_info.get('StudyDate', 'Unknown')}")
    c.drawString(40, height - 160, f"Additional Notes: {patient_info.get('Notes', 'N/A')}")
    c.drawString(40, height - 200, f"Prediction: {label}")
    c.drawString(40, height - 220, f"Probability (Pneumonia): {predicted_prob:.3f}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ----------------------------------------------------
# Endpoint: Predict + Generate PDF
# ----------------------------------------------------
@app.post("/predict/")
async def predict_dicom(
    file: UploadFile = File(...),
    patient_name: str = Form("Unknown"),
    patient_id: str = Form("Unknown"),
    notes: str = Form("N/A")
):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot save uploaded file: {e}")

    try:
        # Auto-detect DICOM vs normal image
        if imghdr.what(tmp_path):
            # Normal image (JPG, PNG)
            img = Image.open(tmp_path).convert("RGB").resize((224, 224))
            img = np.array(img) / 255.0
            ds = None
        else:
            # DICOM
            img, ds = read_dicom_as_image(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    img_batch = np.expand_dims(img, axis=0)
    prob = float(model.predict(img_batch)[0, 0])
    label = "Pneumonia" if prob >= 0.5 else "Normal"

    # Collect metadata
    patient_info = {
        "PatientName": patient_name if patient_name != "Unknown" else getattr(ds, "PatientName", "Unknown") if ds else "Unknown",
        "PatientID": patient_id if patient_id != "Unknown" else getattr(ds, "PatientID", "Unknown") if ds else "Unknown",
        "StudyDate": getattr(ds, "StudyDate", "Unknown") if ds else "Unknown",
        "Notes": notes
    }

    pdf_buffer = make_pdf_report(patient_info, prob, label)
    return StreamingResponse(
        pdf_buffer,
        media_type='application/pdf',
        headers={"Content-Disposition": "attachment; filename=report.pdf"}
    )

# ----------------------------------------------------
# Run server
# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("predict_api:app", host="0.0.0.0", port=8000, reload=True)


# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import StreamingResponse
# import uvicorn
# import numpy as np  
# import io
# import tempfile
# import pydicom
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import cv2

# app = FastAPI()

# MODEL_PATH = "saved_models/densenet_pneumonia.h5"
# model = load_model(MODEL_PATH)

# # --- GRAD-CAM FUNCTION ---
# def generate_gradcam(model, img_array, layer_name):
#     grad_model = tf.keras.models.Model(
#         [model.inputs], 
#         [model.get_layer(layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, 0]

#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
#     heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
#     return heatmap.numpy()

# def overlay_heatmap(heatmap, image, alpha=0.4):
#     heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
#     return overlayed

# # --- PDF REPORT WITH HEATMAP ---
# def make_pdf_report(patient_info, predicted_prob, label, gradcam_image):
#     buffer = io.BytesIO()
#     c = canvas.Canvas(buffer, pagesize=letter)
#     width, height = letter

#     # Header
#     c.setFont("Helvetica-Bold", 16)
#     c.drawString(40, height - 60, "Pneumonia Detection Report")

#     # Patient Info
#     c.setFont("Helvetica", 12)
#     y = height - 100
#     for key in ["PatientName", "PatientID", "StudyDate"]:
#         c.drawString(40, y, f"{key}: {patient_info.get(key, 'Unknown')}")
#         y -= 20

#     # Prediction Results
#     c.drawString(40, y - 10, f"Prediction: {label}")
#     c.drawString(40, y - 30, f"Probability (Pneumonia): {predicted_prob:.3f}")

#     # Insert Grad-CAM Image
#     img_reader = ImageReader(gradcam_image)
#     c.drawImage(img_reader, 40, 150, width=500, height=350)

#     c.setFont("Helvetica-Oblique", 10)
#     c.drawString(40, 130, "Grad-CAM visualization: Red areas indicate regions influencing the model's decision")

#     c.showPage()
#     c.save()
#     buffer.seek(0)
#     return buffer

# # --- MAIN PREDICT ENDPOINT ---
# @app.post("/predict/")
# async def predict_dicom(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
#             tmp.write(contents)
#             tmp_path = tmp.name
#         ds = pydicom.dcmread(tmp_path, force=True)
#         if 'TransferSyntaxUID' not in ds.file_meta:
#             ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
#         ds.decompress()
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Cannot read DICOM file: {e}")

#     # Extract metadata
#     patient_info = {tag: getattr(ds, tag, 'Unknown') for tag in ['PatientName', 'PatientID', 'StudyDate']}

#     # Decode pixel data
#     try:
#         img = ds.pixel_array
#         img = cv2.resize(img, (224, 224))
#         if len(img.shape) == 2:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         img_norm = img / 255.0
#         img_batch = np.expand_dims(img_norm, axis=0)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

#     # Model prediction
#     prob = float(model.predict(img_batch)[0, 0])
#     label = "Pneumonia" if prob >= 0.5 else "Normal"

#     # Generate Grad-CAM
#     last_conv_layer = model.layers[-3].name  # Adjust if needed
#     heatmap = generate_gradcam(model, img_batch, last_conv_layer)
#     gradcam_overlay = overlay_heatmap(heatmap, np.uint8(img))

#     # Save Grad-CAM to memory
#     _, gradcam_png = cv2.imencode('.png', gradcam_overlay)
#     gradcam_bytes = io.BytesIO(gradcam_png)

#     # Create PDF report
#     pdf_buffer = make_pdf_report(patient_info, prob, label, gradcam_bytes)

#     return StreamingResponse(
#         pdf_buffer,
#         media_type='application/pdf',
#         headers={"Content-Disposition": "attachment; filename=report.pdf"}
#     )

# if __name__ == "__main__":
#     uvicorn.run("predict_api:app", host="0.0.0.0", port=8000, reload=True)
