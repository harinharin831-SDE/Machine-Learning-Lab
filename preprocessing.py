"""
=============================================================================
KIDNEY DISEASE IMAGE CLASSIFICATION USING MACHINE LEARNING
Module: Image Preprocessing Pipeline
=============================================================================
This module implements the complete digital image processing pipeline for
kidney ultrasound scans:
  1. Image loading & validation (Handles PIL, NumPy, bytes, and file paths)
  2. Spatial standardization (Resizing to 128 x 128 pixels)
  3. Color space conversion (RGB to Grayscale)
  4. Noise filtering (Gaussian Blur filter)
  5. Contrast enhancement (Histogram Equalization)
  6. Min-Max intensity normalization ([0.0, 1.0] float32)

Note: For academic and research purposes only.
=============================================================================
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Union, Dict, Tuple, Any


TARGET_IMAGE_SIZE = (128, 128)  # Width, Height


def load_image_safely(
    image_input: Union[str, bytes, bytearray, io.BytesIO, Image.Image, np.ndarray]
) -> np.ndarray:
    """
    Safely reads any input format into a valid RGB NumPy array.

    Args:
        image_input: File path, byte stream, PIL Image, or NumPy array.

    Returns:
        np.ndarray: Image array in RGB color space (uint8).

    Raises:
        ValueError: If the image cannot be read or is invalid.
    """
    if image_input is None:
        raise ValueError("Provided image input is None.")

    # If it's already a numpy array
    if isinstance(image_input, np.ndarray):
        img = image_input.copy()
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    # If it's a PIL Image
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        return np.array(img)

    # If it's a file path string
    if isinstance(image_input, str):
        # Use cv2.imdecode with np.fromfile to support Unicode / Windows file paths cleanly
        try:
            with open(image_input, "rb") as f:
                img_bytes = f.read()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if bgr_img is None:
                raise ValueError(f"Could not decode image from path: {image_input}")
            return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Fallback using PIL
            try:
                pil_img = Image.open(image_input).convert("RGB")
                return np.array(pil_img)
            except Exception as pil_err:
                raise ValueError(f"Failed to read image '{image_input}': {str(e)} | {str(pil_err)}")

    # If it's bytes or BytesIO
    if isinstance(image_input, (bytes, bytearray, io.BytesIO)):
        if isinstance(image_input, io.BytesIO):
            image_input.seek(0)
            data = image_input.read()
        else:
            data = image_input

        img_array = np.frombuffer(data, dtype=np.uint8)
        bgr_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if bgr_img is not None:
            return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        # Fallback to PIL
        try:
            pil_img = Image.open(io.BytesIO(data)).convert("RGB")
            return np.array(pil_img)
        except Exception as e:
            raise ValueError(f"Failed to decode image from byte stream: {str(e)}")

    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def preprocess_image(
    image_input: Union[str, bytes, bytearray, io.BytesIO, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    gaussian_kernel_size: Tuple[int, int] = (5, 5),
    gaussian_sigma: float = 0
) -> Dict[str, Any]:
    """
    Executes the 5-stage image preprocessing pipeline:
      1. Original RGB loading
      2. Spatial resizing to 128x128
      3. Grayscale conversion
      4. Gaussian Blur noise reduction
      5. Histogram Equalization contrast enhancement
      6. Min-Max normalization [0, 1]

    Args:
        image_input: Input image in any supported format.
        target_size: Desired resolution (width, height), default (128, 128).
        gaussian_kernel_size: Kernel dimensions for Gaussian blur, default (5, 5).
        gaussian_sigma: Standard deviation for Gaussian kernel.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'original': Loaded RGB array (original dimensions)
            - 'resized': Resized 128x128 RGB array
            - 'gray': 128x128 single-channel grayscale (uint8)
            - 'denoised': 128x128 smoothed grayscale (uint8)
            - 'enhanced': 128x128 contrast-enhanced grayscale (uint8)
            - 'normalized': 128x128 normalized float32 array in [0.0, 1.0]
            - 'normalized_uint8': 128x128 normalized array mapped back to [0, 255] uint8
            - 'final': 128x128 preprocessed image for feature extraction (enhanced uint8)
    """
    # 1. Load Original RGB Image
    original_rgb = load_image_safely(image_input)

    # 2. Resize to Target Size (128x128) using bilinear interpolation
    resized_rgb = cv2.resize(original_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # 3. Grayscale Conversion (Y = 0.299R + 0.587G + 0.114B)
    gray_image = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)

    # 4. Noise Reduction using Gaussian Blur
    # Kernels must be positive odd integers
    ksize = (
        gaussian_kernel_size[0] if gaussian_kernel_size[0] % 2 != 0 else gaussian_kernel_size[0] + 1,
        gaussian_kernel_size[1] if gaussian_kernel_size[1] % 2 != 0 else gaussian_kernel_size[1] + 1
    )
    denoised_image = cv2.GaussianBlur(gray_image, ksize, gaussian_sigma)

    # 5. Contrast Enhancement using Histogram Equalization
    # Stretches intensity range across all 256 gray levels for clear ultrasound tissue boundaries
    enhanced_image = cv2.equalizeHist(denoised_image)

    # 6. Min-Max Intensity Normalization to [0.0, 1.0]
    normalized_float = enhanced_image.astype(np.float32) / 255.0

    # Also provide uint8 normalized image for easy display
    normalized_uint8 = np.clip(normalized_float * 255.0, 0, 255).astype(np.uint8)

    return {
        "original": original_rgb,
        "resized": resized_rgb,
        "gray": gray_image,
        "denoised": denoised_image,
        "enhanced": enhanced_image,
        "normalized": normalized_float,
        "normalized_uint8": normalized_uint8,
        "final": enhanced_image  # uint8 grayscale suited for HOG and GLCM
    }


def calculate_image_histogram(image_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the 256-bin intensity histogram for a grayscale image.

    Args:
        image_gray: Grayscale 2D array (uint8).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (counts, bin_edges)
    """
    hist, bin_edges = np.histogram(image_gray, bins=256, range=(0, 256))
    return hist, bin_edges
