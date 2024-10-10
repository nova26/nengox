import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

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

        print("Creating ensembles")

        positive_ensemble = nengo.Ensemble(
            dvs_process.height * dvs_process.width,
            1,
            neuron_type=nengo.SpikingRectifiedLinear(),
            gain=nengo.dists.Choice([101]),
            bias=nengo.dists.Choice([0]),
        )

        negative_ensemble = nengo.Ensemble(
            dvs_process.height * dvs_process.width,
            1,
            neuron_type=nengo.SpikingRectifiedLinear(),
            gain=nengo.dists.Choice([101]),
            bias=nengo.dists.Choice([0]),
        )

        print("Connection ensembles to DVS")

        nengo.Connection(u[1:: dvs_process.polarity], positive_ensemble.neurons, transform=1.0)
        nengo.Connection(u[0:: dvs_process.polarity], negative_ensemble.neurons, transform=1.0)

        probes = [nengo.Probe(negative_ensemble.neurons), nengo.Probe(positive_ensemble.neurons),nengo.Probe(u)]

    print("Running sim")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(t_length)

    sim_t = sim.trange()
    shape = (len(sim_t), dvs_process.height, dvs_process.width)
    output_spikes_neg = sim.data[probes[0]].reshape(shape) * sim.dt
    output_spikes_pos = sim.data[probes[1]].reshape(shape) * sim.dt

    dt_frame = 0.01
    t_frames = dt_frame * np.arange(int(round(t_length / dt_frame)))

    fig = plt.figure()
    imgs = []
    frame_count = 0
    for t_frame in t_frames:
        frame_count+=1
        t0 = t_frame
        t1 = t_frame + dt_frame
        m = (sim_t >= t0) & (sim_t < t1)

        frame_img = np.zeros((dvs_process.height, dvs_process.width))
        frame_img -= output_spikes_neg[m].sum(axis=0)
        frame_img += output_spikes_pos[m].sum(axis=0)

        if np.abs(frame_img).max() != 0 :
            frame_img = frame_img / np.abs(frame_img).max()


        img = plt.imshow(frame_img, vmin=-1, vmax=1, animated=True)
        imgs.append([img])

    ani = ArtistAnimation(fig, imgs, interval=50, blit=True)

    output_file = os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data', 'model_out.mp4')
    ani.save(output_file, writer='ffmpeg')

    print(f"Total frames {frame_count}")