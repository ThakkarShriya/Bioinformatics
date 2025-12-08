import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy import ndimage as ndi

def apply_image_processing(img):
    # If RGB (3 channels), convert to grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        img = img.astype(np.float32)
        img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    elif img.ndim == 2:
        img = img.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    img /= 255.0  # Normalize to [0, 1]

    # Apply brain mask
    mask = img > 0.1
    img *= mask

    # Smooth with Gaussian blur
    img_smooth = ndi.gaussian_filter(img, sigma=1)

    # Sharpen with Laplacian
    laplace_kernel = np.array([[0, -1, 0],
                                [-1, 4, -1],
                                [0, -1, 0]])
    laplace = ndi.convolve(img_smooth, laplace_kernel)
    img_sharp = np.clip(img_smooth + laplace, 0, 1)

    # Feature detection (Sobel)
    gx = ndi.sobel(img_sharp, axis=0)
    gy = ndi.sobel(img_sharp, axis=1)
    gradient_mag = np.hypot(gx, gy)

    # Spatial transformation (2D rotation)
    theta = np.deg2rad(5)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
    coords = np.indices(img.shape).reshape(2, -1)
    coords_centered = coords - np.array(img.shape).reshape(2, 1) / 2
    rotated_coords = rot_matrix @ coords_centered
    rotated_coords += np.array(img.shape).reshape(2, 1) / 2
    rotated_coords = rotated_coords.reshape(2, *img.shape)

    img_rotated = ndi.map_coordinates(gradient_mag, rotated_coords, order=1, mode='reflect')

    return img_rotated

def show_sample_images(folder_path, title, num_images=5):
    """
    Displays processed MRI slices after masking, filtering, sharpening, rotation.
    """
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith('.gif')])
    plt.figure(figsize=(15, 3))

    for i, filename in enumerate(filenames[:num_images]):
        img_path = os.path.join(folder_path, filename)
        raw_img = imageio.imread(img_path)
        processed_img = apply_image_processing(raw_img)

        plt.subplot(1, num_images, i + 1)
        plt.imshow(processed_img, cmap='gray')
        plt.axis('off')
        plt.title(f"{filename[:10]}...")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def load_images_from_folder(folder_path):
    """
    Loads MRI slices from a folder, applies preprocessing and flattens each image into a vector.
    This preserves explicit linear algebra structure: images → row vectors in data matrix.
    """
    image_vectors = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.gif'):
            path = os.path.join(folder_path, filename)
            img = imageio.imread(path)
            processed = apply_image_processing(img)
            vectorized = processed.flatten()  # 2D matrix to 1D vector
            image_vectors.append(vectorized)

    return np.array(image_vectors)  # Matrix of shape (n_samples, n_pixels)


# === Display processed samples ===
show_sample_images("dataset/healthy", " Processed Healthy Brain Samples")
show_sample_images("dataset/mci", " Processed MCI Brain Samples")


