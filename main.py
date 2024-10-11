import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from nengo_extras.vision import Gabor
import nengo
import nengo_loihi
from scipy.ndimage import gaussian_filter  # Import Gaussian filter

# All NengoLoihi models should call this before model construction
nengo_loihi.set_defaults()

rng = np.random.RandomState(0)

t_length = 20
dt_frame = 0.01
events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\syntactic_left.events'


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


# Function to create and return the neural ensembles
def create_ensembles(dvs_process):
    neurons_per_pixel = dvs_process.height * dvs_process.width

    positive_ensemble = nengo.Ensemble(
        neurons_per_pixel, 1,
        neuron_type=nengo.SpikingRectifiedLinear(),
        gain=nengo.dists.Choice([101]),
        bias=nengo.dists.Choice([0])
    )

    negative_ensemble = nengo.Ensemble(
        neurons_per_pixel, 1,
        neuron_type=nengo.SpikingRectifiedLinear(),
        gain=nengo.dists.Choice([101]),
        bias=nengo.dists.Choice([0])
    )

    gabor_ensemble = nengo.Ensemble(
        neurons_per_pixel, 1,
        neuron_type=nengo.RectifiedLinear(),
    )

    gaussian_ensemble = nengo.Ensemble(
        neurons_per_pixel, 1,
        neuron_type=nengo.RectifiedLinear(),
    )

    return positive_ensemble, negative_ensemble, gabor_ensemble, gaussian_ensemble


# Function to visualize the neural activity and save as an MP4 file
def visualize_and_save(output, shape, sim_t, t_frames, file_name):
    fig = plt.figure()
    imgs = []
    frame_count = 0

    for t_frame in t_frames:
        frame_count += 1
        t0 = t_frame
        t1 = t_frame + dt_frame
        m = (sim_t >= t0) & (sim_t < t1)

        frame_img = np.zeros(shape[1:])
        frame_img += output[m].sum(axis=0)

        if np.abs(frame_img).max() != 0:
            frame_img = frame_img / np.abs(frame_img).max()

        img = plt.imshow(frame_img, vmin=-1, vmax=1, animated=True)
        imgs.append([img])

    ani = ArtistAnimation(fig, imgs, interval=50, blit=True)
    ani.save(file_name, writer='ffmpeg')

    print(f"Total frames {frame_count} for {file_name}")


if __name__ == '__main__':
    model = nengo.Network(label="NEF summary")

    with model:
        print("Creating DVS object")
        dvs_process = nengo_loihi.dvs.DVSFileChipProcess(
            file_path=events_file_name, channels_last=True, dvs_height=20, dvs_width=50,
        )
        u = nengo.Node(dvs_process)

        print("Creating ensembles")
        positive_ensemble, negative_ensemble, gabor_ensemble, gaussian_ensemble = create_ensembles(dvs_process)

        print("Connecting ensembles to DVS")
        nengo.Connection(u[1:: dvs_process.polarity], positive_ensemble.neurons, transform=1.0)
        nengo.Connection(u[0:: dvs_process.polarity], negative_ensemble.neurons, transform=1.0)

        # Gabor filter connection
        gabor_bank = Gabor().generate(1, (11, 11), rng)  # Single Gabor filter for convolution
        gabor_transform_matrix = gabor_transform(dvs_process.height, dvs_process.width, gabor_bank)
        nengo.Connection(positive_ensemble.neurons, gabor_ensemble.neurons, transform=gabor_transform_matrix)

        # Gaussian filter connection
        gaussian_transform_matrix_values = gaussian_transform_matrix(dvs_process.height, dvs_process.width, sigma=1.0)
        nengo.Connection(positive_ensemble.neurons, gaussian_ensemble.neurons, transform=gaussian_transform_matrix_values)

        probes = [nengo.Probe(negative_ensemble.neurons),
                  nengo.Probe(positive_ensemble.neurons),
                  nengo.Probe(gabor_ensemble.neurons),
                  nengo.Probe(gaussian_ensemble.neurons)
                  ]

    print("Running simulation")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(t_length)

    sim_t = sim.trange()
    shape = (len(sim_t), dvs_process.height, dvs_process.width)
    output_spikes_neg = sim.data[probes[0]].reshape(shape) * sim.dt
    output_spikes_pos = sim.data[probes[1]].reshape(shape) * sim.dt
    gabor_filtered_output = sim.data[probes[2]].reshape(shape) * sim.dt
    gaussian_filtered_output = sim.data[probes[3]].reshape(shape) * sim.dt

    t_frames = dt_frame * np.arange(int(round(t_length / dt_frame)))

    # Visualize and save input ensemble output
    visualize_and_save(output_spikes_pos - output_spikes_neg, shape, sim_t, t_frames,
                       os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data', 'model_out_input.mp4'))

    # Visualize and save Gabor-filtered output
    visualize_and_save(gabor_filtered_output, shape, sim_t, t_frames,
                       os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data', 'gabor_filtered_model_out.mp4'))

    # Visualize and save Gaussian-filtered output
    visualize_and_save(gaussian_filtered_output, shape, sim_t, t_frames,
                       os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data', 'gaussian_filtered_model_out.mp4'))
