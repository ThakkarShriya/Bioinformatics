"""
PCA of Volume Maps Hypothesis

1. Project Name: Early Detection of Dementia Using PCA-Based Structural Biomarkers from Brain MRI Slices

2. Project description:

This project presents a learning activity that enables students to apply eigenvalue-based dimensionality
reduction (PCA) to investigate a practical biomedical problem. We demonstrate how to process 2D brain
MRI slices from healthy individuals and early-stage dementia patients, and describe how students can
extract pixel intensity data using open-source Python libraries. We then illustrate how to analyze this
data using principal component analysis (PCA) to uncover structural biomarkers—such as cortical thinning
or ventricle enlargement—that are associated with Mild Cognitive Impairment (MCI), an early stage of
dementia. This hands-on activity enhances student motivation and prepares students to apply linear
algebra and statistical methods to real-world clinical imaging challenges.

3. Who might this project be good for?

The Early Detection of Dementia Using PCA-Based Structural Biomarkers project is designed to excite
students majoring in Biomedical Engineering, Neuroscience, Computer Science, Data Science, Applied
Mathematics, or any student who would like to learn how to translate medical images into mathematical
evidence for disease detection. One way to figure out if this project might give you a competitive
advantage in your academic career is to ask yourself the following questions:

a. Am I required to take courses like Bio 1A/1B/1C, Math 2B, Math 22, Engr 11 or C S 3A/3B/3C at
   Foothill College or a different institution?
b. Do I have an interest in using computers to analyze brain scans or other clinical data to detect
   diseases early?
c. Do I plan to pursue a career in healthcare, biotechnology, or AI where medical data and coding
   intersect?

If your answer to these questions is yes, then this project offers a powerful introduction to the
process of extracting meaningful features from real medical images, modeling patterns using linear
algebra (PCA), and testing hypotheses with statistical tools. This project provides a hands-on
foundation in biomedical image processing, dimensionality reduction, and quantitative clinical
research.

4. Project resources:




5. Project Outcomes:

Beginning-Level Outcomes:
By the end of this project, you will be able to extract features from brain MRI slices and use PCA to identify
structural differences between healthy individuals and early dementia patients. You will learn to apply a t-test to
PCA scores to verify if the differences are statistically significant.

More-Advanced Outcomes:
You might extend the project by including more principal components, reconstructing MRI slices from PCA outputs, or
analyzing 3D scans. You could also explore how brain structure changes over time using longitudinal data.

"""
# Hypothesis: The distributions of principal-component (PC) scores derived from 2-D brain-MRI slices differ
# significantly between individuals with Mild Cognitive Impairment (MCI) and cognitively healthy controls.

# Null Hypothesis (H0): The average PCA scores for MCI and healthy individuals are the same for all principal
# components used (e.g., PC1 to PC5). That is, there is no statistically significant difference in brain structure
# between the two groups based on these PCA-derived features.


import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from numpy.linalg import svd

def show_sample_images(folder_path, title, num_images=5):
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import os

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
def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.gif'):
            path = os.path.join(folder_path, filename)
            img = imageio.imread(path)
            img = img.astype(np.float32) / 255.0  # Normalize pixel values
            images.append(img.flatten())  # Flatten to 1D
    return np.array(images)

show_sample_images("dataset/healthy", "Healthy Brain Samples", num_images=5)
show_sample_images("dataset/mci", "MCI Brain Samples", num_images=5)
def perform_pca(data_matrix, num_components=5):
    mean_vector = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - mean_vector
    U, S, Vt = svd(centered_data, full_matrices=False)
    pcs = Vt[:num_components]
    scores = centered_data @ pcs.T
    return scores, pcs

mci_path = "dataset/mci"
healthy_path = "dataset/healthy"

mci_data = load_images_from_folder(mci_path)
healthy_data = load_images_from_folder(healthy_path)
all_data = np.vstack([mci_data, healthy_data])

scores, pcs = perform_pca(all_data, num_components=5)

mci_scores = scores[:len(mci_data)]
healthy_scores = scores[len(mci_data):]

print("\n🎯 T-Test Results on PCA Scores:\n")
for i in range(5):
    t_stat, p_val = ttest_ind(mci_scores[:, i], healthy_scores[:, i])
    print(f"PC{i+1}: t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("→ Statistically significant difference.\n")
    else:
        print("→ Not statistically significant.\n")

# === Plot PC1 vs PC2 ===
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


