import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
from gradients import load_gradient_table

# FID loader
def load_fid(bruker_dir):
    dic, data = ng.bruker.read(bruker_dir)
    data = ng.bruker.remove_digital_filter(dic, data)
    data = np.ravel(data).astype(np.complex128)
    return data, dic

# SPRITE reconstruction
def sprite_recon_from_fid(data, gY, gZ, N=32, npoint=32, start_point=2):
    Data = data[start_point::npoint]

    n_grad = len(gY)
    n_data = len(Data)

    if n_data > n_grad:
        print(f"{n_data - n_grad}")
        Data = Data[:n_grad]

    ky = np.floor(gY * N / 2 + N / 2).astype(int)
    kz = np.floor(gZ * N / 2 + N / 2).astype(int)

    K = np.zeros((N, N), dtype=np.complex128)
    W = np.zeros((N, N), dtype=int)

    for i in range(n_grad):
        y, z = ky[i], kz[i]
        if 0 <= y < N and 0 <= z < N:
            K[y, z] += Data[i]
            W[y, z] += 1

    mask = W > 0
    K[mask] /= W[mask]

    return K, W, Data

# Centered zero-fill
def pad_kspace_centered(K, Nzf):
    N = K.shape[0]
    if Nzf < N:
        raise ValueError("Target size must be >= original size")
    
    Kzf = np.zeros((Nzf, Nzf), dtype=K.dtype)

    c_old = N // 2
    c_new = Nzf // 2

    y_slice = slice(c_new - c_old, c_new - c_old + N)
    z_slice = slice(c_new - c_old, c_new - c_old + N)

    Kzf[y_slice, z_slice] = K
    return Kzf

# FFT
def kspace_to_image(K):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(K)))


# Plot k-space trajectory for diagnostics
def plot_kspace_trajectory(gY, gZ, N=32, title="k-space trajectory"):
    # Map to integer indices (like SPRITE reconstruction)
    ky = gY * N / 2
    kz = gZ * N / 2

    plt.figure(figsize=(5,5))
    plt.scatter(ky, kz, s=10, c='blue', alpha=0.6)
    plt.axhline(0, color='k', lw=1)
    plt.axvline(0, color='k', lw=1)
    plt.gca().set_aspect('equal')
    plt.xlabel("ky")
    plt.ylabel("kz")
    plt.title(title)
    plt.grid(True)
    plt.show()



# Main script
fid_dir = "C:/Users/camgu/Documents/NMR/300WB/20251106_1H_phantom_MRI/13"
gY_file = "MRI/gp/2DSPRITE_1inter_726pts_32x32_Y"
gZ_file = "MRI/gp/2DSPRITE_1inter_726pts_32x32_Z"

# Load data
data, dic = load_fid(fid_dir)
gY = load_gradient_table(gY_file)
gZ = load_gradient_table(gZ_file)

# Check normalization
if np.max(np.abs(gY)) > 1.01 or np.max(np.abs(gZ)) > 1.01:
    raise ValueError("Gradients must be normalized to [-1, 1]")

# Reconstruct original SPRITE disk
K, W, DataUsed = sprite_recon_from_fid(data, gY, gZ, N=32, npoint=32, start_point=2)

print("\n--- Original SPRITE Disk Diagnostics ---")
print(f"Data points used: {len(DataUsed)}")
print(f"Gradient points: {len(gY)}")
print(f"Filled k-space points: {np.count_nonzero(W)}")
print(f"Coverage fraction: {np.count_nonzero(W)/(32*32):.3f}")

# Plot original k-space hit count
plt.figure(figsize=(5,4))
plt.imshow(W, origin="lower")
plt.title("Original k-space hit count")
plt.colorbar(label="hits")
plt.tight_layout()
plt.show()

# Centered zero-fill to 128x128
Kzf = pad_kspace_centered(K, Nzf=128)
img_zf = kspace_to_image(Kzf)

# Display side-by-side
img_orig = kspace_to_image(K)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(np.abs(img_orig), cmap="gray")
plt.title("Original 32x32 FFT magnitude")
plt.colorbar()

# Original k-space trajectory
plot_kspace_trajectory(gY, gZ, N=32, title="SPRITE k-space trajectory (32x32 mapping)")

# If you want, also for the zero-filled 128x128 grid:
plot_kspace_trajectory(gY, gZ, N=128, title="SPRITE k-space trajectory (128x128 mapping)")


plt.subplot(1,2,2)
plt.imshow(np.abs(img_zf), cmap="gray")
plt.title("Zero-filled 128x128 FFT magnitude (centered)")
plt.colorbar()
plt.tight_layout()
plt.show()

print("\n--- Zero-filled k-space Diagnostics ---")
print(f"Padded k-space shape: {Kzf.shape}")
print(f"Filled points after padding: {np.count_nonzero(Kzf)}")
