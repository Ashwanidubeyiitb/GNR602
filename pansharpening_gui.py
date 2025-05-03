import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from tqdm import tqdm
import os

class PansharpeningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Satellite Image Pansharpening")
        self.root.geometry("1200x800")
        
        # Set default paths for New York dataset
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newyork")
        self.pan_path = tk.StringVar(value=os.path.join(base_path, "LC08_L1TP_013032_20250323_20250331_02_T1_B8.TIF"))
        self.ms_paths = [
            tk.StringVar(value=os.path.join(base_path, "LC08_L1TP_013032_20250323_20250331_02_T1_B2.TIF")),  # Blue
            tk.StringVar(value=os.path.join(base_path, "LC08_L1TP_013032_20250323_20250331_02_T1_B3.TIF")),  # Green
            tk.StringVar(value=os.path.join(base_path, "LC08_L1TP_013032_20250323_20250331_02_T1_B4.TIF"))   # Red
        ]
        self.method = tk.StringVar(value="IHS")
        self.contrast_factor = tk.DoubleVar(value=1.0)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Left panel for controls
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # File selection
        ttk.Label(control_frame, text="Panchromatic Image:").pack(anchor=tk.W)
        ttk.Entry(control_frame, textvariable=self.pan_path, width=40).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Browse", command=lambda: self.browse_file(self.pan_path)).pack(pady=2)
        
        ttk.Label(control_frame, text="Multispectral Bands:").pack(anchor=tk.W, pady=(10,0))
        for i, band in enumerate(["Blue", "Green", "Red"]):
            ttk.Label(control_frame, text=f"{band} Band:").pack(anchor=tk.W)
            ttk.Entry(control_frame, textvariable=self.ms_paths[i], width=40).pack(fill=tk.X, pady=2)
            ttk.Button(control_frame, text="Browse", 
                      command=lambda idx=i: self.browse_file(self.ms_paths[idx])).pack(pady=2)
        
        # Method selection
        ttk.Label(control_frame, text="Pansharpening Method:").pack(anchor=tk.W, pady=(10,0))
        ttk.Radiobutton(control_frame, text="IHS", variable=self.method, value="IHS").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="PCA", variable=self.method, value="PCA").pack(anchor=tk.W)
        
        # Contrast adjustment
        ttk.Label(control_frame, text="Contrast Factor:").pack(anchor=tk.W, pady=(10,0))
        ttk.Scale(control_frame, from_=0.5, to=2.0, variable=self.contrast_factor, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        # Process button
        ttk.Button(control_frame, text="Process", command=self.process_image).pack(pady=20)
        
        # Right panel for visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def browse_file(self, path_var):
        filename = filedialog.askopenfilename(
            filetypes=[("TIFF files", "*.tif")],
            initialdir=os.path.dirname(path_var.get())
        )
        if filename:
            path_var.set(filename)
            
    def enhance_contrast(self, image, factor):
        # Convert to float32 for calculations
        img_float = image.astype(np.float32)
        
        # Normalize to [0, 1]
        img_norm = (img_float - np.min(img_float)) / (np.max(img_float) - np.min(img_float) + 1e-6)
        
        # Apply contrast adjustment
        img_enhanced = np.clip((img_norm - 0.5) * factor + 0.5, 0, 1)
        
        # Convert back to original range
        return (img_enhanced * (np.max(img_float) - np.min(img_float)) + np.min(img_float)).astype(image.dtype)
    
    def rgb_to_ihs(self, rgb):
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        I = (R + G + B) / 3.0
        min_val = np.minimum(np.minimum(R, G), B)
        S = 1 - (3 * min_val / (R + G + B + 1e-6))
        numerator = 0.5 * ((R - G) + (R - B))
        denominator = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-6
        H = np.arccos(numerator / denominator)
        H[B > G] = (2 * np.pi) - H[B > G]
        H = H / (2 * np.pi)
        return np.stack((I, H, S), axis=-1)
    
    def ihs_to_rgb(self, ihs):
        I, H, S = ihs[:, :, 0], ihs[:, :, 1], ihs[:, :, 2]
        H = H * 2 * np.pi
        R = np.zeros_like(I)
        G = np.zeros_like(I)
        B = np.zeros_like(I)
        
        idx1 = (H < 2*np.pi/3)
        idx2 = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
        idx3 = (H >= 4*np.pi/3)
        
        R[idx1] = I[idx1] * (1 + (S[idx1] * np.cos(H[idx1])) / (np.cos(np.pi/3 - H[idx1]) + 1e-6))
        B[idx1] = I[idx1] * (1 - S[idx1])
        G[idx1] = 3*I[idx1] - (R[idx1] + B[idx1])
        
        H2 = H[idx2] - 2*np.pi/3
        R[idx2] = I[idx2] * (1 - S[idx2])
        G[idx2] = I[idx2] * (1 + (S[idx2] * np.cos(H2)) / (np.cos(np.pi/3 - H2) + 1e-6))
        B[idx2] = 3*I[idx2] - (R[idx2] + G[idx2])
        
        H3 = H[idx3] - 4*np.pi/3
        G[idx3] = I[idx3] * (1 - S[idx3])
        B[idx3] = I[idx3] * (1 + (S[idx3] * np.cos(H3)) / (np.cos(np.pi/3 - H3) + 1e-6))
        R[idx3] = 3*I[idx3] - (G[idx3] + B[idx3])
        
        rgb = np.stack((R, G, B), axis=-1)
        return np.clip(rgb, 0, 65535)
    
    def process_image(self):
        try:
            # Check if files exist
            for path in [self.pan_path.get()] + [p.get() for p in self.ms_paths]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
            
            # Load images
            with rasterio.open(self.pan_path.get()) as pan_ds:
                pan = pan_ds.read(1).astype(np.float32)
                pan_transform = pan_ds.transform
                pan_profile = pan_ds.profile
            
            ms_bands = []
            for path in [p.get() for p in self.ms_paths]:
                with rasterio.open(path) as ms_ds:
                    ms_data = ms_ds.read(
                        1,
                        out_shape=(pan.shape[0], pan.shape[1]),
                        resampling=Resampling.bilinear
                    ).astype(np.float32)
                    ms_bands.append(ms_data)
            
            ms_upsampled = np.stack(ms_bands, axis=-1)
            
            # Apply pansharpening
            if self.method.get() == "IHS":
                result = self.process_ihs(pan, ms_upsampled)
            else:
                result = self.process_pca(pan, ms_upsampled)
            
            # Enhance contrast
            contrast_factor = self.contrast_factor.get()
            result_enhanced = np.zeros_like(result)
            for i in range(3):
                result_enhanced[:, :, i] = self.enhance_contrast(result[:, :, i], contrast_factor)
            
            # Display results
            self.ax1.clear()
            self.ax2.clear()
            
            # Original RGB (downsampled for display)
            rgb_display = ms_upsampled[::4, ::4, :]
            self.ax1.imshow(rgb_display.astype(np.uint16))
            self.ax1.set_title('Original RGB')
            self.ax1.axis('off')
            
            # Pansharpened result
            self.ax2.imshow(result_enhanced.astype(np.uint16))
            self.ax2.set_title(f'{self.method.get()} Pansharpened')
            self.ax2.axis('off')
            
            self.canvas.draw()
            
            # Save result
            save_path = filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("TIFF files", "*.tif")],
                initialdir=os.path.dirname(self.pan_path.get())
            )
            if save_path:
                out_profile = pan_profile.copy()
                out_profile.update({
                    'height': result.shape[0],
                    'width': result.shape[1],
                    'count': 3,
                    'dtype': 'uint16'
                })
                
                with rasterio.open(save_path, 'w', **out_profile) as dst:
                    for i in range(3):
                        dst.write(result_enhanced[:, :, i].astype('uint16'), i + 1)
                
                messagebox.showinfo("Success", f"Image saved successfully to {save_path}")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def process_ihs(self, pan, ms_upsampled):
        height, width = pan.shape
        result = np.zeros((height, width, 3), dtype=np.float32)
        tile_size = 512
        
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                
                tile_rgb = ms_upsampled[y:y_end, x:x_end, :]
                tile_pan = pan[y:y_end, x:x_end]
                
                ihs = self.rgb_to_ihs(tile_rgb)
                ihs[:, :, 0] = tile_pan
                rgb_out = self.ihs_to_rgb(ihs)
                
                result[y:y_end, x:x_end, :] = rgb_out
        
        return result
    
    def process_pca(self, pan, ms_upsampled):
        height, width, bands = ms_upsampled.shape
        ms_flat = ms_upsampled.reshape(-1, bands)
        
        pca = PCA(n_components=bands)
        pcs = pca.fit_transform(ms_flat)
        
        pan_flat = pan.flatten()
        pan_scaled = (pan_flat - np.min(pan_flat)) / (np.max(pan_flat) - np.min(pan_flat) + 1e-6)
        pc1_scaled = (pcs[:, 0] - np.min(pcs[:, 0])) / (np.max(pcs[:, 0]) - np.min(pcs[:, 0]) + 1e-6)
        
        pan_scaled = (pan_scaled - np.mean(pan_scaled)) / (np.std(pan_scaled) + 1e-6)
        pc1_scaled = (pc1_scaled - np.mean(pc1_scaled)) / (np.std(pc1_scaled) + 1e-6)
        
        pcs[:, 0] = pan_scaled
        
        ms_reconstructed = pca.inverse_transform(pcs)
        result = ms_reconstructed.reshape(height, width, bands)
        
        return result

if __name__ == "__main__":
    root = tk.Tk()
    app = PansharpeningGUI(root)
    root.mainloop() 