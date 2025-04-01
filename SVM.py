import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from google.colab import files

# -----------------------------
# Visualization Utility
# -----------------------------
def visualize_preprocessing_steps(image, processed_images, titles):
    """Display different preprocessing stages of an image."""
    plt.figure(figsize=(20, 10))
    for i, (img, title) in enumerate(zip(processed_images, titles)):
        plt.subplot(1, len(processed_images), i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()

# -----------------------------
# Step 1: Remove UI Elements
# -----------------------------
def remove_text_bars_signs(image):
    """Mask out UI elements (text, bars, borders) from ultrasound image."""
    processed = image.copy()
    h, w = processed.shape[:2]
    cv2.rectangle(processed, (0, 0), (w, int(0.1*h)), (0,0,0), -1)       # Top
    cv2.rectangle(processed, (0, int(0.9*h)), (w, h), (0,0,0), -1)       # Bottom
    cv2.rectangle(processed, (0, 0), (int(0.09*w), h), (0,0,0), -1)      # Left
    cv2.rectangle(processed, (int(0.95*w), 0), (w, h), (0,0,0), -1)      # Right
    return processed

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
def preprocess_image(image_path):
    """Apply image enhancement pipeline."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    img_clean = remove_text_bars_signs(img)
    img_resized = cv2.resize(img_clean, (800, 600))
    img_contrast = cv2.convertScaleAbs(img_resized, alpha=2, beta=50)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_sharpened = cv2.filter2D(img_contrast, -1, kernel)

    visualize_preprocessing_steps(img, [img_clean, img_resized, img_sharpened],
                                  ["Text Removed", "Resized", "Sharpened"])
    return img_sharpened

# -----------------------------
# Step 3: Radial Mask
# -----------------------------
def mask_borders(image):
    """Mask circular region of interest in center."""
    mask = np.zeros_like(image)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    cv2.circle(mask, center, radius, (255,255,255), -1)
    return cv2.bitwise_and(image, image, mask=mask[:,:,0])

# -----------------------------
# Step 4: Hyperechoic Mask
# -----------------------------
def create_hyperechoic_mask(image):
    contrast = cv2.convertScaleAbs(image, alpha=2.0, beta=50)
    _, binary = cv2.threshold(contrast, 200, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

# -----------------------------
# Step 5: Clean Hyperechoic Mask
# -----------------------------
def clean_hyperechoic_mask(mask):
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    total_area = mask.shape[0] * mask.shape[1]
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > total_area * 0.002:
            cleaned[labels == i] = 255
    return cleaned

# -----------------------------
# Step 6: Geometric Features
# -----------------------------
def extract_geometric_features(contour):
    area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    distances = [np.sqrt((px - cx)**2 + (py - cy)**2) for px, py in np.squeeze(contour)]
    roundness = np.std(distances)

    pca = PCA(n_components=2)
    linearity = pca.fit(np.squeeze(contour)).explained_variance_ratio_[0]

    sharpness = 0
    for i in range(2, len(contour)-2):
        p1, p2, p3 = contour[i-2][0], contour[i][0], contour[i+2][0]
        angle = abs(np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0]))
        angle = np.degrees(angle)
        if 20 <= angle <= 80:
            sharpness += 1

    return {'area': area, 'centroid': (cx, cy), 'roundness': roundness, 'linearity': linearity, 'sharpness': sharpness}

# -----------------------------
# Step 7: Grayscale Features
# -----------------------------
def extract_grayscale_features(contour, image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    pixels = cv2.bitwise_and(image, image, mask=mask)[mask > 0]
    return {'echogenicity': np.mean(pixels), 'echo_variability': np.std(pixels)}

# -----------------------------
# Step 8: Rule-Based Labeling
# -----------------------------
def rule_based_labeling(geo, gray):
    if geo['area'] < 100 or geo['roundness'] > 15 or gray['echogenicity'] > 100:
        return 'negative'
    if geo['linearity'] < 0.8 or geo['sharpness'] < 3:
        return 'negative'
    return 'positive'

# -----------------------------
# Step 9: Frame Processing
# -----------------------------
def process_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features, labels = [], []
    for c in contours:
        geo = extract_geometric_features(c)
        gray_feats = extract_grayscale_features(c, gray)
        if geo and gray_feats:
            label = rule_based_labeling(geo, gray_feats)
            feat = [geo['area'], geo['centroid'][0], geo['centroid'][1], geo['roundness'],
                    geo['linearity'], geo['sharpness'], gray_feats['echogenicity'], gray_feats['echo_variability']]
            features.append(feat)
            labels.append(1 if label == 'positive' else 0)
    return features, labels

# -----------------------------
# Step 10: SVM Training
# -----------------------------
def train_model(video_paths):
    features_all, labels_all = [], []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            feats, labels = process_frame(frame)
            if feats:
                features_all.extend(feats)
                labels_all.extend(labels)
        cap.release()

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(features_all))
    y = np.array(labels_all)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = SVC(kernel='rbf', C=1, gamma='auto')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    return clf, scaler

# -----------------------------
# Step 11: Inference
# -----------------------------
def predict_free_fluid_in_video(path, clf, scaler):
    cap = cv2.VideoCapture(path)
    detected, total = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        total += 1
        feats, _ = process_frame(frame)
        if feats:
            feats_norm = scaler.transform(feats)
            if np.any(clf.predict(feats_norm) == 1):
                detected += 1
    cap.release()
    print(f"Free fluid in {detected}/{total} frames.")
    return detected > 1

# -----------------------------
# Main Execution Block (Colab)
# -----------------------------
uploaded = files.upload()
video_paths = list(uploaded.keys())
svm_model, scaler = train_model(video_paths)

uploaded = files.upload()
new_video_path = next(iter(uploaded))
if predict_free_fluid_in_video(new_video_path, svm_model, scaler):
    print("Free fluid is detected.")
else:
    print("No free fluid detected.")
