import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Load 2D Bruker ser file
def load_2d_ser(bruker_dir):
    dic, data = ng.bruker.read(bruker_dir)

    # Remove digital filter
    # data = ng.bruker.remove_digital_filter(dic, data)
    # Ensure complex dtype
    data = data.astype(np.complex128)
    # Print shape for sanity
    print("Raw_shape:", data.shape)
    # fig = px.scatter_3d(data)
    # fig.show()

    return data, dic


def cos_win(n, power=1, shift=0.0):
    t = np.linspace(0, np.pi, n)
    w = np.sin(shift * np.pi + (1 - shift) * t)

    return w**power


def ap(data, power=2, shift=0.3):
    n1, n2 = data.shape
    w1 = cos_win(n1, power=power, shift=shift)
    w2 = cos_win(n2, power=power, shift=shift)

    return data*w1[:, None]*w2[None, :]


# Zero filling
def zero_fill(data, zf1=None, zf2=None):
    n1, n2 = data.shape
    zf1 = zf1 or n1
    zf2 = zf2 or n2
    out = np.zeros((zf1, zf2), dtype=data.dtype)
    out[:n1, :n2] = data

    return out


# 2D FFT
def fft2c(data):
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(data)
        )
    )


# 3D Visualization
def plot_3d_surface(data, title="3D Surface Plot", cmap="viridis"):
    """Plot 3D surface view of 2D data using matplotlib"""
    n1, n2 = data.shape
    x = np.arange(n2)
    y = np.arange(n1)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data, cmap=cmap, alpha=0.9)
    ax.set_xlabel("Frequency Axis 2")
    ax.set_ylabel("Frequency Axis 1")
    ax.set_zlabel("Magnitude")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5)
    plt.tight_layout()
    return fig


def plot_3d_surface_plotly(data, title="3D Surface Plot"):
    """Plot 3D surface view of 2D data using plotly (interactive)"""
    n1, n2 = data.shape
    fig = go.Figure(data=[go.Surface(z=data)])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Frequency Axis 2",
            yaxis_title="Frequency Axis 1",
            zaxis_title="Magnitude"
        ),
        width=1000,
        height=800
    )
    return fig


# Main
bruker_dir = "C:/Users/camgu/Documents/NMR/300WB/20250520_7Li_phantom_MRI/13/pdata/1"

# Load data
data, dic = load_2d_ser(bruker_dir)

# data = ap(data, power=3, shift=0.1)
data_zf = zero_fill(data, zf1=128, zf2=512)
# data_zf = data

# FT
img = fft2c(data_zf)
img_mag = np.abs(img)

# Plots
plt.figure(figsize=(6,5))
plt.imshow(img_mag, cmap="gray", origin="lower", aspect="auto")
# plt.title("2D FFT Magnitude")
# plt.xlabel()
# plt.ylabel()
# plt.colorbar()
# plt.xlim(25, 150)
plt.tight_layout()
# plt.show()

# 3D Surface Plot - Interactive (Plotly)
# fig_3d = plot_3d_surface_plotly(img_mag, title="3D FFT Magnitude")
# fig_3d.show()

# Optional: 3D Surface Plot - Matplotlib (static)
fig_3d_mpl = plot_3d_surface(data, title="3D Magnitude", cmap="hot")
plt.show()
