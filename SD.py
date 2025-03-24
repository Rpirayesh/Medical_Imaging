import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from skimage.segmentation import chan_vese
import matplotlib
matplotlib.use('Agg')


# ========== Set video path here ==========
video_path = "/home/rezapirayesh/Files/TamuG/TX/Texas/med/vids/2.mp4"  # <<< REPLACE THIS
assert os.path.exists(video_path), f"Video path does not exist: {video_path}"

# ========== Utility Functions ==========

def remove_text_bars_signs(image):
    processed_image = image.copy()
    height, width = processed_image.shape[:2]
    cv2.rectangle(processed_image, (0, 0), (width, int(0.1 * height)), (0, 0, 0), -1)  # Top
    cv2.rectangle(processed_image, (0, 0), (int(0.09 * width), height), (0, 0, 0), -1)  # Left
    cv2.rectangle(processed_image, (int(0.95 * width), 0), (width, height), (0, 0, 0), -1)  # Right
    return processed_image

def enhance_contrast(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img_gray)

def denoise_image(img):
    return cv2.fastNlMeansDenoising(img, None, 30, 7, 21)

def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def preprocess_image(image):
    img_cleaned = remove_text_bars_signs(image)
    img_resized = cv2.resize(img_cleaned, (800, 600))
    img_contrast = cv2.convertScaleAbs(img_resized, alpha=2, beta=50)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharpen = cv2.filter2D(img_contrast, -1, kernel)
    img_eq = enhance_contrast(img_sharpen)
    img_denoised = denoise_image(img_eq)
    img_thresh = adaptive_threshold(img_denoised)
    return img_sharpen

def apply_chan_vese(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    img_float = img_eq.astype(float) / 255.0
    cv_result = chan_vese(img_float, mu=0.1, lambda1=0.8, lambda2=0.8,
                          tol=1e-4, max_num_iter=800, dt=0.25)
    return (cv_result * 255).astype(np.uint8)

def segment_free_fluid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

def highlight_edges(img, contour_color=(0, 255, 0), thickness=3, min_area=600):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(contour_img, [cnt], -1, contour_color, thickness)
    return contour_img

# ========== Load CLIP Model ==========
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")

text_labels = ["Normal Ultrasound", "Free Fluid Present"]
location_queries = [
    "Free fluid in the upper part", "Free fluid in the lower part",
    "Free fluid on the left", "Free fluid on the right", "Free fluid in the center"
]

# ========== Process Video ==========
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(text=text_labels, images=pil_image, return_tensors="pt", padding=True)
    probs = model(**inputs).logits_per_image.softmax(dim=1).squeeze()

    location_inputs = processor(text=location_queries, images=pil_image, return_tensors="pt", padding=True)
    location_probs = model(**location_inputs).logits_per_image.softmax(dim=1).squeeze()
    best_location = location_queries[torch.argmax(location_probs)]

    preprocessed = preprocess_image(frame)
    segmented_chanvese = highlight_edges(apply_chan_vese(preprocessed))
    segmented_morph = highlight_edges(segment_free_fluid(frame))

    # Visualization
    fig, axs = plt.subplots(1, 5, figsize=(20, 8), dpi=150)

    axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Frame", fontsize=14)
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Preprocessed Frame", fontsize=14)
    axs[1].axis("off")

    axs[2].imshow(segmented_morph, cmap="gray")
    axs[2].set_title("Segmentation (Threshold + Morph)", fontsize=14)
    axs[2].axis("off")

    axs[3].imshow(segmented_chanvese, cmap="gray")
    axs[3].set_title("Segmentation (Chan-Vese)", fontsize=14)
    axs[3].axis("off")

    title_text = f"{text_labels[0]}: {probs[0]:.2%}\n{text_labels[1]}: {probs[1]:.2%}\nLocation: {best_location}"
    axs[4].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[4].set_title("Classification & Location\n" + title_text, fontsize=14)
    axs[4].axis("off")

    plt.tight_layout()
    plt.savefig(f"frame_{frame_count}.png")
    plt.close()


cap.release()
