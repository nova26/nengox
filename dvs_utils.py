

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import nengo


# Function to create and return the transformation matrix for Gabor filtering
def gabor_transform(height, width, gabor_bank):
    transform = np.zeros((height * width, height * width))
    filter_size = gabor_bank.shape[1]
    half_filter = filter_size // 2

    for i in range(height):
        for j in range(width):
            neuron_idx = i * width + j
            for di in range(-half_filter, half_filter + 1):
                for dj in range(-half_filter, half_filter + 1):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        pixel_idx = ni * width + nj
                        filter_val = gabor_bank[0, di + half_filter, dj + half_filter]
                        transform[neuron_idx, pixel_idx] = filter_val
    return transform


# Function to create a Gaussian transform matrix
def gaussian_transform_matrix(height, width, sigma=1.0):
    transform = np.zeros((height * width, height * width))

    # Define the Gaussian kernel based on the sigma
    size = int(6 * sigma)  # Gaussian kernel size: +/- 3 sigma
    if size % 2 == 0:
        size += 1  # Ensure odd size for proper centering

    # Create a 2D Gaussian kernel
    gaussian_kernel = np.fromfunction(
        lambda x, y: np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize

    # Apply the Gaussian kernel across the transform matrix
    half_size = size // 2
    for i in range(height):
        for j in range(width):
            neuron_idx = i * width + j
            for di in range(-half_size, half_size + 1):
                for dj in range(-half_size, half_size + 1):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        pixel_idx = ni * width + nj
                        filter_val = gaussian_kernel[di + half_size, dj + half_size]
                        transform[neuron_idx, pixel_idx] = filter_val

    return transform


def create_input_ensembles(neurons_per_pixel):

    positive_ensemble = nengo.Ensemble(
        neurons_per_pixel,
        1,
        neuron_type=nengo.SpikingRectifiedLinear(),
        gain=nengo.dists.Choice([101]),
        bias=nengo.dists.Choice([0])
    )

    negative_ensemble = nengo.Ensemble(
        neurons_per_pixel,
        1,
        neuron_type=nengo.SpikingRectifiedLinear(),
        gain=nengo.dists.Choice([101]),
        bias=nengo.dists.Choice([0])
    )

    return positive_ensemble, negative_ensemble


# Function to create and return the neural ensembles
def create_filter_ensembles(neurons_per_pixel):
    gabor_ensemble = nengo.Ensemble(
        neurons_per_pixel, 1,
        neuron_type=nengo.RectifiedLinear(),
    )

    gaussian_ensemble = nengo.Ensemble(
        neurons_per_pixel, 1,
        neuron_type=nengo.RectifiedLinear(),
    )

    return gabor_ensemble, gaussian_ensemble


