import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import os
import imghdr

def load_image(path, target_size=(224, 224)):
    """
    Load an image from path — handles both DICOM and normal image formats.
    Returns a normalized numpy array (H, W, 3) in range [0,1].
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        # Check if it's a normal image (jpg, png)
        if imghdr.what(path) in ['jpeg', 'png']:
            img = Image.open(path).convert("RGB").resize(target_size)
            img = np.array(img) / 255.0
            return img

        # Else assume it's DICOM
        ds = pydicom.dcmread(path, force=True)
        # Handle missing Transfer Syntax UID
        if "TransferSyntaxUID" not in ds.file_meta:
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        data = apply_voi_lut(ds.pixel_array, ds)
        data = data.astype(float)
        data -= np.min(data)
        data /= np.max(data)
        data = np.stack([data]*3, axis=-1)  # grayscale -> RGB
        img = Image.fromarray((data * 255).astype(np.uint8)).resize(target_size)
        img = np.array(img) / 255.0
        return img
    except Exception as e:
        raise ValueError(f"Error loading image {path}: {e}")
