import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import ArtistAnimation


import nengo
import nengo_loihi

# All NengoLoihi models should call this before model construction
nengo_loihi.set_defaults()

rng = np.random.RandomState(0)

gain = 101

t_length = 1

events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\dvs-from-file-events.events'

if __name__ == '__main__':

    with nengo.Network() as net:
        dvs_process = nengo_loihi.dvs.DVSFileChipProcess(
            file_path=events_file_name, channels_last=True,dvs_height = 480,dvs_width = 640,
        )

        u = nengo.Node(dvs_process)

        ensembles = [
            nengo.Ensemble(
               dvs_process.height * dvs_process.width,
                1,
                neuron_type=nengo.SpikingRectifiedLinear(),
                gain=nengo.dists.Choice([gain]),
                bias=nengo.dists.Choice([0]),
            )
            for _ in range(dvs_process.polarity)
        ]

        print(f"Number of ensembles {len(ensembles)}")

        for k, e in enumerate(ensembles):
            u_channel = u[k :: dvs_process.polarity]
            nengo.Connection(u_channel, e.neurons, transform=1.0)

        probes = [nengo.Probe(e.neurons) for e in ensembles]

    print("Running sim")
    with nengo_loihi.Simulator(net) as sim:
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
        #frame_img = frame_img / np.abs(frame_img).max()

        img = plt.imshow(frame_img, vmin=-1, vmax=1, animated=True)
        imgs.append([img])

    ani = ArtistAnimation(fig, imgs, interval=50, blit=True)
    ani.save(r"C:\Users\USER\PycharmProjects\nengox\data\model_out.mp4", writer='ffmpeg')  # Or use 'imagemagick' for .gif
