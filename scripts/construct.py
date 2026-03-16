import numpy as np
import nmrglue as ng
from pprint import pprint

# Bruker FID
def load_fid(bruker_dir):
    """
    Load Bruker FID/SER using nmrglue.
    Returns:
        data : 1D complex numpy array
        dic  : Bruker dictionary
    """
    dic, data = ng.bruker.read(bruker_dir, pprog_file='pulseprogram')
    # pprint(dic)
    # print(type(dic), len(dic))
    pprint(data)
    print(type(data), len(data))

    # Remove digital filter (removes initial redundant points)
    # data = ng.bruker.remove_digital_filter(dic, data)
    
    data = data[69:23300] # Perfect bounds
    # data = data[69:20000]



    # Flatten
    # data = np.ravel(data).astype(np.complex128)

    return data, dic

def load_ascii(asc_file):
    """
    Docstring for load_ascii
    
    :param asc_file: Description
    """
    A = 0
    dic = ''
    try:
        # with open(asc_file, 'r', encoding='ascii') as file:
        #     A = file.read()
        A = np.loadtxt(asc_file, delimiter=',')
    except UnicodeDecodeError:
        print(f"Error reading file: {asc_file} is not a valid ASCII file.")
    except FileNotFoundError:
        print(f"Error: The file {asc_file} was not found.")
    
    # Save real/imag as separate arrays
    r = np.array([val for val in A[0::2, 1]])
    i = np.array([val for val in A[1::2, 1]])

    data = np.empty(r.shape, dtype=np.complex128)
    data.real = r
    data.imag = i

    pprint(data)
    return data, dic

# Reconstruct
def sprite_recon_from_fid(data, gY, gZ, N=32, npoint=32, start_point=2, threshold_frac=0.05):
    Data = data[start_point::npoint]

    n_grad = len(gY)
    n_data = len(Data)

    if n_data > n_grad:
        print(f"Truncating {n_data - n_grad} trailing points")
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


    # Threshold small magnitudes
    if threshold_frac is not None:
        thresh = threshold_frac * np.max(np.abs(K))
        K[np.abs(K) < thresh] = 0

    # Diagnostics
    info = {
        "n_fid_points": len(data),
        "n_data_used": n_data,
        "n_gradients": n_grad,
        "unique_ky": len(np.unique(ky)),
        "unique_kz": len(np.unique(kz)),
        "filled_kspace_points": np.count_nonzero(W),
        "coverage_fraction": np.count_nonzero(W) / (N * N),
        "max_hits_per_point": W.max(),
    }

    return K, W, Data, info

# Pad k-space to grid
def pad_kspace_centered(K, W, Nzf):
    N = K.shape[0]

    Kzf = np.zeros((Nzf, Nzf), dtype=K.dtype)
    Wzf = np.zeros((Nzf, Nzf), dtype=W.dtype)

    c_old = N // 2
    c_new = Nzf // 2

    y_slice = slice(c_new - c_old, c_new - c_old + N)
    z_slice = slice(c_new - c_old, c_new - c_old + N)

    Kzf[y_slice, z_slice] = K
    Wzf[y_slice, z_slice] = W

    return Kzf, Wzf



# Zero fill
def zero_fill_kspace(K, W, Nzf):
    N = K.shape[0]
    if Nzf < N:
        raise ValueError("Nzf must be >= original size")

    Kzf = np.zeros((Nzf, Nzf), dtype=K.dtype)
    Wzf = np.zeros((Nzf, Nzf), dtype=W.dtype)

    c0 = Nzf // 2
    c1 = N // 2

    sl = slice(c0 - c1, c0 - c1 + N)

    Kzf[sl, sl] = K
    Wzf[sl, sl] = W

    return Kzf, Wzf


# FFT
def kspace_to_image(K):
    return np.fft.ifftshift(
        np.fft.fft2(
            np.fft.fftshift(K)
        )
    )
    # return np.fft.fft2(np.fft.fftshift(K))



import matplotlib.pyplot as plt
from gradients import load_gradient_table

# Load
fid_dir = "C:/Users/camgu/Documents/NMR/300WB/20251106_1H_phantom_MRI/12"
fid_file = 'MRI/data/fid/2D_matlab_ascii.txt'
gY_file = "MRI/gp/2DSPRITE_1inter_726pts_32x32_Y"
gZ_file = "MRI/gp/2DSPRITE_1inter_726pts_32x32_Z"

# data, dic = load_fid(fid_dir)
data, dic = load_ascii(fid_file)
gY = load_gradient_table(gY_file)
gZ = load_gradient_table(gZ_file)

# Plot FID
plt.figure(figsize=(5, 4))
plt.plot(data)
plt.title("FID")
plt.tight_layout()
# plt.show()

# Gradient sanity check
if np.max(np.abs(gY)) > 1.01 or np.max(np.abs(gZ)) > 1.01:
    raise ValueError("SPRITE gradients must be normalized to [-1, 1]")

# Build
K, W, dataUsed, info = sprite_recon_from_fid(
    data,
    gY,
    gZ,
    N=32,
    npoint=32,
    start_point=5,
    threshold_frac=0.0,
)

# plt.figure(figsize=(5, 4))
# plt.imshow(np.abs(K), cmap="viridis")
# plt.title("K-space")
# plt.colorbar()
# plt.tight_layout()
# plt.show()

# Diagnostics
print("\n--- SPRITE Reconstruction Diagnostics ---")
for k, v in info.items():
    print(f"{k:>25s} : {v}")


# K-space coverage plot
# plt.figure(figsize=(5, 4))
# plt.imshow(W, origin="lower")
# plt.title("k-space hit count")
# plt.colorbar(label="hits")
# plt.tight_layout()
# plt.show()


# Apply zero fill/FFT
# Kzf = zero_fill_kspace(K, Nzf=64)
# img = kspace_to_image(Kzf)

# img = kspace_to_image(K)
Kzf, Wzf = pad_kspace_centered(K, W, Nzf=64)
img = kspace_to_image(Kzf)

plt.figure(figsize=(5, 4))
plt.imshow(Wzf, origin="lower")
plt.title("Zero-filled k-space hit count")
plt.colorbar(label="hits")
plt.tight_layout()

plt.figure(figsize=(5, 4))
plt.imshow(np.abs(Kzf), cmap="viridis")
plt.title("Zero-filled K-space")
plt.colorbar()
plt.tight_layout()

plt.figure(figsize=(5, 4))
plt.imshow(np.abs(img), cmap="viridis")
plt.title("Img")
plt.colorbar()
plt.tight_layout()
plt.show()
