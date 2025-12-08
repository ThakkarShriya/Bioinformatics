import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from numpy.linalg import svd
from sklearn.metrics import mean_absolute_error

# ========== Function to visualize sample images ==========
def show_sample_images(folder_path, title, num_images=5):
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith('.gif')])
    plt.figure(figsize=(15, 3))
    for i, filename in enumerate(filenames[:num_images]):
        img_path = os.path.join(folder_path, filename)
        img = imageio.imread(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"{filename[:10]}...")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ========== Load and preprocess images ==========
def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.gif'):
            path = os.path.join(folder_path, filename)
            img = imageio.imread(path)
            img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
            images.append(img.flatten())  # Flatten image to 1D vector
    return np.array(images)

# ========== PCA Implementation ==========
def perform_pca(data_matrix, num_components=5):
    mean_vector = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - mean_vector
    U, S, Vt = svd(centered_data, full_matrices=False)
    pcs = Vt[:num_components]  # Principal components
    scores = centered_data @ pcs.T  # Project data onto PCs
    return scores, pcs

# ========== Show sample images ==========
show_sample_images("dataset/healthy", "Healthy Brain Samples")
show_sample_images("dataset/mci", "MCI Brain Samples")

# ========== Load and prepare datasets ==========
mci_path = "dataset/mci"
healthy_path = "dataset/healthy"

mci_data = load_images_from_folder(mci_path)
healthy_data = load_images_from_folder(healthy_path)

all_data = np.vstack([mci_data, healthy_data])  # Combine both groups

# ========== Run PCA ==========
scores, pcs = perform_pca(all_data, num_components=5)

# Separate PCA scores
mci_scores = scores[:len(mci_data)]
healthy_scores = scores[len(mci_data):]

# ========== T-Tests ==========
print("\n🎯 T-Test Results on PCA Scores:\n")
for i in range(5):
    t_stat, p_val = ttest_ind(mci_scores[:, i], healthy_scores[:, i])
    print(f"PC{i+1}: t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("→ Statistically significant difference.\n")
    else:
        print("→ Not statistically significant.\n")

mean_mci = np.mean(mci_scores, axis=0)
mean_healthy = np.mean(healthy_scores, axis=0)
mae = mean_absolute_error(mean_mci, mean_healthy)

print(f"📊 Mean Absolute Error between MCI and Healthy PCA means: {mae:.4f}\n")

# ========== PCA Visualization ==========
plt.figure(figsize=(8, 6))
plt.scatter(healthy_scores[:, 0], healthy_scores[:, 1], label='Healthy', alpha=0.7)
plt.scatter(mci_scores[:, 0], mci_scores[:, 1], label='MCI', alpha=0.7)
plt.title("PCA of Brain MRI Slices")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


