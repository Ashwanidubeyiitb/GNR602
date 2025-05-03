import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import rasterio
from tqdm import tqdm

# --- Load PAN and MS Images ---

pan_path = r"C:\Users\Dell\Desktop\newyork\LC08_L1TP_013032_20250323_20250331_02_T1_B8.TIF"
ms_paths = [
    r"C:\Users\Dell\Desktop\newyork\LC08_L1TP_013032_20250323_20250331_02_T1_B2.TIF",
    r"C:\Users\Dell\Desktop\newyork\LC08_L1TP_013032_20250323_20250331_02_T1_B3.TIF",
    r"C:\Users\Dell\Desktop\newyork\LC08_L1TP_013032_20250323_20250331_02_T1_B4.TIF"
]

with rasterio.open(pan_path) as pan_ds:
    pan = pan_ds.read(1).astype(np.float32)
    pan_transform = pan_ds.transform
    pan_profile = pan_ds.profile

ms_bands = []
for path in ms_paths:
    with rasterio.open(path) as ms_ds:
        ms_data = ms_ds.read(
            1,
            out_shape=(pan.shape[0], pan.shape[1]),
            resampling=rasterio.enums.Resampling.bilinear
        ).astype(np.float32)
        ms_bands.append(ms_data)

ms_upsampled = np.stack(ms_bands, axis=-1)

print(f"Full image size for processing: {pan.shape}")

# --- PCA Pansharpening on Full Image ---

# Flatten full MS for PCA
height_full, width_full, bands = ms_upsampled.shape
ms_flat = ms_upsampled.reshape(-1, bands)

print("Starting full-image PCA Pansharpening...")

# Progress bar for PCA process
with tqdm(total=1, desc="PCA Processing Full Image") as pbar:
    pca = PCA(n_components=bands)
    pcs = pca.fit_transform(ms_flat)

    pan_flat = pan.flatten()
    pan_scaled = (pan_flat - np.min(pan_flat)) / (np.max(pan_flat) - np.min(pan_flat) + 1e-6)
    pc1_scaled = (pcs[:, 0] - np.min(pcs[:, 0])) / (np.max(pcs[:, 0]) - np.min(pcs[:, 0]) + 1e-6)

    pan_scaled = (pan_scaled - np.mean(pan_scaled)) / (np.std(pan_scaled) + 1e-6)
    pc1_scaled = (pc1_scaled - np.mean(pc1_scaled)) / (np.std(pc1_scaled) + 1e-6)

    pcs[:, 0] = pan_scaled

    ms_reconstructed = pca.inverse_transform(pcs)
    ms_reconstructed = ms_reconstructed.reshape(height_full, width_full, bands)

    pbar.update(1)

pca_pansharpened = np.clip(ms_reconstructed, 0, 65535).astype(np.uint16)

print("✅ Full-image PCA Pansharpening completed!")

# --- Plot Final Result (downsampled for plotting) ---

#plt.figure(figsize=(10, 10))
#plt.imshow(pca_pansharpened[::20, ::20, :])  # Downsample for display
#plt.title('Full Image PCA-based Pansharpened')
#plt.axis('off')
#plt.show()

# --- Save Final Image ---

save_path = r"C:\Users\Dell\Desktop\newyork\pca_pansharpened_full.tif"

out_profile = pan_profile.copy()
out_profile.update({
    'height': pca_pansharpened.shape[0],
    'width': pca_pansharpened.shape[1],
    'count': 3,
    'dtype': 'uint16'
})

with rasterio.open(save_path, 'w', **out_profile) as dst:
    dst.write(pca_pansharpened[:, :, 2], 1)
    dst.write(pca_pansharpened[:, :, 1], 2)
    dst.write(pca_pansharpened[:, :, 0], 3)

print(f"✅ Saved full-image PCA pansharpened result at {save_path}")
