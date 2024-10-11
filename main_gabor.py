import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from nengo_extras.vision import Gabor, Mask  # Import Gabor filters from nengo_extras
from nengo_extras.matplotlib import tile

import nengo
import nengo_loihi

# All NengoLoihi models should call this before model construction
nengo_loihi.set_defaults()

rng = np.random.RandomState(0)
t_length = 20

events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\syntactic_left.events'

if __name__ == '__main__':
    model = nengo.Network(label="NEF summary")

    with model:
        print("Creating DVS object")
        dvs_process = nengo_loihi.dvs.DVSFileChipProcess(
            file_path=events_file_name, channels_last=True, dvs_height=20, dvs_width=50,
        )

        u = nengo.Node(dvs_process)

        print("Creating input ensemble")

        # Each neuron corresponds to one pixel, so the dimensionality is 1
        input_ensemble = nengo.Ensemble(
            dvs_process.height * dvs_process.width,  # Number of neurons (one per pixel)
            1,  # Dimensionality is 1 (each neuron outputs a single value)
            neuron_type=nengo.SpikingRectifiedLinear(),
            gain=nengo.dists.Choice([101]),
            bias=nengo.dists.Choice([0]),
        )

        print("Connecting input ensemble to DVS")

        # Connect DVS output to input ensemble
        nengo.Connection(u[1:: dvs_process.polarity], input_ensemble.neurons, transform=1.0)

        print("Applying Gabor filters")


        # Apply Gabor filters like a convolution, where each neuron receives input from a small patch
        def gabor_transform(height, width, gabor_bank):
            # Create a transformation matrix that applies the Gabor filter to each neuron's input
            transform = np.zeros((height * width, height * width))

            filter_size = gabor_bank.shape[1]  # Assuming square Gabor filters
            half_filter = filter_size // 2

            for i in range(height):
                for j in range(width):
                    neuron_idx = i * width + j  # Index of the neuron corresponding to pixel (i, j)

                    # Apply Gabor filter centered on this neuron
                    for di in range(-half_filter, half_filter + 1):
                        for dj in range(-half_filter, half_filter + 1):
                            ni = i + di
                            nj = j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                pixel_idx = ni * width + nj  # Index of neighboring pixel
                                filter_val = gabor_bank[0, di + half_filter, dj + half_filter]
                                transform[neuron_idx, pixel_idx] = filter_val

            return transform


        # Generate Gabor filters
        gabor_bank = Gabor().generate(1, (11, 11), rng)  # Single Gabor filter for convolution

        # Create a new ensemble to represent the output after Gabor filtering
        gabor_ensemble = nengo.Ensemble(
            dvs_process.height * dvs_process.width,  # One neuron per pixel
            1,  # Dimensionality is 1 (one output per neuron)
            neuron_type=nengo.RectifiedLinear(),
        )

        # Apply Gabor filter as a transformation between input and Gabor ensembles
        gabor_transform_matrix = gabor_transform(dvs_process.height, dvs_process.width, gabor_bank)
        nengo.Connection(input_ensemble.neurons, gabor_ensemble.neurons, transform=gabor_transform_matrix)

        # Add a probe to capture the Gabor-filtered output
        gabor_probe = nengo.Probe(gabor_ensemble.neurons)

    print("Running simulation")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(t_length)

    # Retrieve the probed data
    gabor_filtered_output = sim.data[gabor_probe]

    # Reshape the data for visualization
    sim_t = sim.trange()
    shape = (len(sim_t), dvs_process.height, dvs_process.width)
    gabor_filtered_output = gabor_filtered_output.reshape(shape) * sim.dt

    dt_frame = 0.01
    t_frames = dt_frame * np.arange(int(round(t_length / dt_frame)))

    fig = plt.figure()
    imgs = []
    frame_count = 0

    for t_frame in t_frames:
        frame_count += 1
        t0 = t_frame
        t1 = t_frame + dt_frame
        m = (sim_t >= t0) & (sim_t < t1)

        # Sum the Gabor-filtered output over the current time frame
        frame_img = np.zeros((dvs_process.height, dvs_process.width))
        frame_img += gabor_filtered_output[m].sum(axis=0)

        # Normalize the image for visualization
        if np.abs(frame_img).max() != 0:
            frame_img = frame_img / np.abs(frame_img).max()

        # Visualize the Gabor-filtered output as an image
        img = plt.imshow(frame_img, vmin=-1, vmax=1, animated=True)
        imgs.append([img])

    # Create animation (no individual frame saving)
    ani = ArtistAnimation(fig, imgs, interval=50, blit=True)

    # Save the MP4 output directly
    output_file = os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data', 'gabor_filtered_model_out.mp4')
    ani.save(output_file, writer='ffmpeg')

    print(f"Total frames {frame_count}")
