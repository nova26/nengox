import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from nengo_extras.vision import Gabor, Mask
import nengo
import nengo_loihi
from dvs_utils import create_input_ensembles

nengo_loihi.set_defaults()
rng = np.random.RandomState(42)  # Change this to any valid seed between 0 and 2**32 - 1

t_length = 1
dt_frame = 0.01
left_events_file_name = '/home/avi/projects/nengox/data/thun_00_a_left.events'
right_events_file_name = '/home/avi/projects/nengox/data/thun_00_a_right.events'

dvs_height = 3
dvs_width = 600
if __name__ == '__main__':
    model = nengo.Network(label="NEF summary")

    with model:
        print("Creating DVS object")
        left_dvs_process = nengo_loihi.dvs.DVSFileChipProcess(
            file_path=left_events_file_name, channels_last=True, dvs_height=dvs_height, dvs_width=dvs_width,
        )

        u_left = nengo.Node(left_dvs_process)

        print("Creating input ensembles to DVS")
        left_positive_ensemble, left_negative_ensemble = create_input_ensembles(
            left_dvs_process.height * left_dvs_process.width
        )

        print("Connecting input ensembles to DVS")
        nengo.Connection(
            u_left[0:: left_dvs_process.polarity], left_negative_ensemble.neurons, transform=1.0
        )
        nengo.Connection(
            u_left[1:: left_dvs_process.polarity], left_positive_ensemble.neurons, transform=1.0
        )

        print("Connecting Gabor filter to input ensembles")
        # Connect the positive ensemble to the Gabor-filtered ensemble

        left_probes = [
            nengo.Probe(left_negative_ensemble.neurons),
            nengo.Probe(left_positive_ensemble.neurons),
        ]

    print("Running simulation")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(t_length)

    sim_t = sim.trange()
    shape = (len(sim_t), left_dvs_process.height, left_dvs_process.width)
    output_spikes_neg = sim.data[left_probes[0]].reshape(shape) * sim.dt
    output_spikes_pos = sim.data[left_probes[1]].reshape(shape) * sim.dt

    # Save the output spikes as .npy files
    np.save('/home/avi/projects/nengox/data/model/sim_t.npy', sim_t)
    np.save('/home/avi/projects/nengox/data/model/output_spikes_neg.npy', output_spikes_neg)
    np.save('/home/avi/projects/nengox/data/model/output_spikes_pos.npy', output_spikes_pos)

    print("Saved output spikes and Gabor output to .npy files.")
