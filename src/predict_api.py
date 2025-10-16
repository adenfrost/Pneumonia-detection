import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import JSONResponse
from PIL import Image
import pydicom
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import tempfile 
import os

# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI(title="Pneumonia Detection with Grad-CAM")

# Serve static files (images generated)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained model
MODEL_PATH = "saved_models/densenet_pneumonia.h5"
model = tf.keras.models.load_model(MODEL_PATH)


# -------------------------------
# Utility: Load image (JPEG or DICOM)
# -------------------------------
def load_image(file: UploadFile):
    if file.filename.lower().endswith(".dcm"):
        ds = pydicom.dcmread(file.file)
        pixel_array = ds.pixel_array.astype(float)
        image = cv2.resize(pixel_array, (224, 224))
        image = cv2.cvtColor(np.uint8(image / np.max(image) * 255), cv2.COLOR_GRAY2RGB)
    else:
        image = Image.open(file.file).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image)

    img_array = image.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return image, img_array

def generate_gradcam(model, img_array, power=1.0):
    """
    Generate Grad-CAM by computing gradient of output with respect to input.
    This method works with ANY model architecture without needing to access internal layers.
    """
    
    img_tensor = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor, training=False)
        
        # Get predicted class
        class_idx = tf.argmax(predictions[0])
        class_channel = predictions[0, class_idx]
    
    # Compute gradients with respect to input image
    grads = tape.gradient(class_channel, img_tensor)
    
    if grads is None:
        raise ValueError("Gradients are None!")
    
    print(f"Predictions: {predictions[0]}")
    print(f"Predicted class: {class_idx} (0=Normal, 1=Pneumonia)")
    print(f"Confidence: {predictions[0, class_idx]:.4f}")
    print(f"Gradients shape: {grads.shape}")
    print(f"Gradients range: [{tf.reduce_min(grads):.6f}, {tf.reduce_max(grads):.6f}]")
    
    # Take absolute value and convert to numpy
    grads = tf.abs(grads)[0]  # Remove batch dimension
    
    # Compute saliency map by taking max across channels
    saliency = tf.reduce_max(grads, axis=-1).numpy()
    
    print(f"Saliency map range: [{np.min(saliency):.6f}, {np.max(saliency):.6f}]")
    
    # Normalize to [0, 1]
    saliency = saliency - np.min(saliency)
    saliency_max = np.max(saliency)
    
    if saliency_max > 0:
        saliency /= saliency_max
    else:
        print("WARNING: Saliency map is all zeros!")
        saliency = np.ones_like(saliency) * 0.5
    
    # Apply power for contrast
    saliency = saliency ** power
    
    # Resize to 224x224 if needed
    if saliency.shape != (224, 224):
        saliency = cv2.resize(saliency, (224, 224))
    
    saliency = np.clip(saliency, 0, 1)
    
    print(f"Final saliency range: [{np.min(saliency):.6f}, {np.max(saliency):.6f}]")
    
    return saliency


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """Overlay saliency heatmap on original image with medical-grade VIRIDIS colormap."""
    
    # Ensure heatmap is in [0, 1] range
    heatmap = np.clip(heatmap, 0, 1)
    
    # Convert to 8-bit for OpenCV colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Use TURBO (medical-grade) or HOT colormap instead of JET
    # TURBO: cooler colors (blue/cyan) for low, warmer (yellow/red) for high - better for medical
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    
    # Convert from BGR to RGB
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Ensure original image is uint8
    original_uint8 = np.uint8(np.clip(original_img * 255, 0, 255))
    
    # Blend images - reduce alpha for more balanced view
    overlay = cv2.addWeighted(
        original_uint8, 
        1 - alpha, 
        heatmap_color, 
        alpha, 
        0
    )
    
    return Image.fromarray(overlay)
# -------------------------------
# Utility: PDF Report Generator
# -------------------------------
def create_pdf_report(original_img_path, heatmap_img_path, prediction_label, probability):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Pneumonia Detection Report</b>", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Prediction: <b>{prediction_label}</b>", styles["Normal"]))
    content.append(Paragraph(f"Confidence: <b>{probability * 100:.2f}%</b>", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Original X-ray:", styles["Heading3"]))
    content.append(RLImage(original_img_path, width=250, height=250))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Grad-CAM Heatmap:", styles["Heading3"]))
    content.append(RLImage(heatmap_img_path, width=250, height=250))

    doc.build(content)
    temp_file.seek(0)
    return temp_file


# -------------------------------
# API Endpoint with UI
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def main_page():
    return """
    <html>
        <head>
            <title>Pneumonia Detection UI</title>
        </head>
        <body style="font-family: Arial; margin: 50px;">
            <h2>Pneumonia Detection with Grad-CAM</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".jpeg,.jpg,.png,.dcm" required>
                <br><br>
                <label>Heatmap Power (contrast):</label>
                <input type="number" step="0.1" name="power" value="2.0" min="0.5" max="5.0">
                <br><br>
                <label>Overlay Alpha (transparency):</label>
                <input type="number" step="0.1" name="alpha" value="0.5" min="0.1" max="1.0">
                <br><br>
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    power: float = Form(2.0),
    alpha: float = Form(0.5)
):
    try:
        # --- Load and preprocess image ---
        original_img, img_array = load_image(file)
        preds = model.predict(img_array)
        prob = float(preds[0][0])
        label = "Pneumonia" if prob > 0.5 else "Normal"

        # --- Generate Grad-CAM heatmap ---
        heatmap = generate_gradcam(model, img_array, power=power)
        heatmap_img = overlay_heatmap(original_img, heatmap, alpha=alpha)

        # --- Save both images ---
        os.makedirs("static", exist_ok=True)
        orig_path = f"static/{file.filename}_orig.jpg"
        heatmap_path = f"static/{file.filename}_heatmap.jpg"
        Image.fromarray(original_img).save(orig_path)
        heatmap_img.save(heatmap_path)

        # --- Create PDF report ---
        pdf = create_pdf_report(orig_path, heatmap_path, label, prob)

        # --- Return result UI (shows images + link) ---
        html_response = f"""
        <html>
        <head><title>Prediction Result</title></head>
        <body style="font-family: Arial; margin: 50px;">
            <h2>Prediction: {label}</h2>
            <h3>Confidence: {prob * 100:.2f}%</h3>
            <div style="display:flex; gap:40px; align-items:center;">
                <div>
                    <h4>Original X-ray</h4>
                    <img src="/{orig_path}" width="300" style="border:1px solid #ddd; border-radius:8px;">
                </div>
                <div>
                    <h4>Grad-CAM Heatmap</h4>
                    <img src="/{heatmap_path}" width="300" style="border:1px solid #ddd; border-radius:8px;">
                </div>
            </div>
            <br><br>
            <a href="/{heatmap_path}" target="_blank">🔍 View Heatmap Fullscreen</a><br><br>
            <a href="/download_pdf?path={pdf.name}" download>📄 Download Report (PDF)</a><br><br>
            <a href="/">⬅ Go Back</a>
        </body>
        </html>
        """
        return HTMLResponse(content=html_response, status_code = 200)

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Error: {str(e)}"})

@app.get("/download_pdf")
def download_pdf(path: str):
    pdf_file = open(path, "rb")
    return StreamingResponse(pdf_file, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=report.pdf"})
