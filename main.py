import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import nengo
import nengo_loihi

# All NengoLoihi models should call this before model construction
nengo_loihi.set_defaults()

rng = np.random.RandomState(0)

t_length = 5

events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\synthetic_events_synthetic.events'

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
            neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
            gain=nengo.dists.Choice([101]),
            bias=nengo.dists.Choice([0]),
        )

        negative_ensemble = nengo.Ensemble(
            dvs_process.height * dvs_process.width,
            1,
            neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
            gain=nengo.dists.Choice([101]),
            bias=nengo.dists.Choice([0]),
        )

        print("Connection ensembles to DVS")


        nengo.Connection(u[0:: dvs_process.polarity], positive_ensemble.neurons, transform=1.0)
        nengo.Connection(u[1:: dvs_process.polarity], negative_ensemble.neurons, transform=1.0)

        probes = [nengo.Probe(negative_ensemble.neurons),nengo.Probe(positive_ensemble.neurons)]

    print("Running sim")
    with nengo.Simulator(model) as sim:
        sim.run(t_length)

    sim_t = sim.trange()
    shape = (len(sim_t), dvs_process.height, dvs_process.width)
    output_spikes_neg = sim.data[probes[0]].reshape(shape) * sim.dt
    output_spikes_pos = sim.data[probes[1]].reshape(shape) * sim.dt

    dt_frame = 0.01
    t_frames = dt_frame * np.arange(int(round(t_length / dt_frame)))

    fig = plt.figure()
    imgs = []
    for t_frame in t_frames:
        t0 = t_frame
        t1 = t_frame + dt_frame
        m = (sim_t >= t0) & (sim_t < t1)

        frame_img = np.zeros((dvs_process.height, dvs_process.width))
        frame_img -= output_spikes_neg[m].sum(axis=0)
        frame_img += output_spikes_pos[m].sum(axis=0)

        min_val = frame_img.min()
        max_val = frame_img.max()

        if max_val != min_val:  # Avoid division by zero if all values are the same
            frame_img = 2 * (frame_img - min_val) / (max_val - min_val) - 1  # Normalize to range [-1, 1]

        img = plt.imshow(frame_img, vmin=-1, vmax=1, animated=True)
        imgs.append([img])

    ani = ArtistAnimation(fig, imgs, interval=50, blit=True)
    ani.save(r"C:\Users\USER\PycharmProjects\nengox\data\model_out.mp4",
             writer='ffmpeg')  # Or use 'imagemagick' for .gif
