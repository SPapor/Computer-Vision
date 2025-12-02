import os
import numpy as np
import shutil
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

main_dir = "data"
res = "кластери_результат"
s = (500, 500)

files = [f for f in os.listdir(main_dir) if f.lower().endswith(('.png'))]

if len(files) < 2:
    exit()

n_clusters = min(6, len(files))

os.makedirs(res, exist_ok=True)


def load_img(f):
    path = os.path.join(main_dir, f)
    img = Image.open(path).convert("RGB").resize(s)
    return np.array(img) / 255.0


def ultimate_features(img):
    hsv_img = Image.fromarray((img * 255).astype('uint8')).convert('HSV')
    hsv = np.array(hsv_img)
    h = hsv[:, :, 0] / 179.0
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    gray = np.mean(img, axis=2)

    features = []

    features.append(((h > 0.2) & (h < 0.45) & (s > 0.4)).mean())

    features.append(((h > 0.5) & (h < 0.72) & (s > 0.35)).mean())

    features.append((s < 0.2).mean())

    brown_gray = ((h > 0.05) & (h < 0.25) | (h > 0.0) & (h < 0.05)) & (s < 0.4) & (v > 0.3)
    features.append(brown_gray.mean())

    features.append((v > 0.93).mean())

    grad_y, grad_x = np.gradient(gray)
    edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
    features.append(edges.mean())
    features.append(edges.std())

    hist, _ = np.histogram(gray, bins=64, range=(0, 1), density=True)
    hist = hist + 0.0000000001
    features.append(entropy(hist))


    features.append((v < 0.3).mean())

    return features


X = []
valid_files = []

for f in files:
    try:
        img = load_img(f)
        feats = ultimate_features(img)
        X.append(feats)
        valid_files.append(f)
    except Exception as e:
        print(f"ПОМИЛКА")

X = np.array(X)
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, n_init=30, random_state=42)
labels = kmeans.fit_predict(X_scaled)

names = [
    "місто та забудова",
    "вода та водойми",
    "поля та сільгоспугіддя",
    "гори та скелі",
    "ліси та густа рослинність",
    "степ, пустирі, пісок"
]

for f, label in zip(valid_files, labels):
    cluster_name = names[label] if label < len(names) else f"інше_{label}"
    folder = f"{res}/кластер_{label}_{cluster_name}"
    os.makedirs(folder, exist_ok=True)
    shutil.copy(os.path.join(main_dir, f), os.path.join(folder, f))

