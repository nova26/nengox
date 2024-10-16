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

dvs_height = 100
dvs_width = 200
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

        print("Creating Gabor filters")
        # Number of Gabor filters (same as number of neurons in the Gabor ensemble)
        n_hid = 1000
        gabor_size = (11, 11)  # Size of each Gabor filter
        gabor_bank = Gabor().generate(left_dvs_process.height * left_dvs_process.width, gabor_size, rng=rng)  # Shape: (n_hid, 11, 11)

        # Use Mask to align the Gabor filters with the (320, 240) input space
        encoders = Mask((dvs_height, dvs_width)).populate(gabor_bank, rng=rng, flatten=True)

        # Create a Gabor ensemble with the generated encoders
        left_positive_gabor_ensemble = nengo.Ensemble(
            n_neurons=left_dvs_process.height * left_dvs_process.width,
            dimensions=left_dvs_process.height * left_dvs_process.width,
            neuron_type=nengo.RectifiedLinear(),
            encoders=encoders
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
        nengo.Connection(
            left_positive_ensemble.neurons,
            left_positive_gabor_ensemble.neurons,
            transform=1.0
        )

        left_probes = [
            nengo.Probe(left_negative_ensemble.neurons),
            nengo.Probe(left_positive_ensemble.neurons),
            nengo.Probe(left_positive_gabor_ensemble.neurons)  # Probe for Gabor-filtered output
        ]

    print("Running simulation")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(t_length)

    sim_t = sim.trange()
    shape = (len(sim_t), left_dvs_process.height, left_dvs_process.width)
    output_spikes_neg = sim.data[left_probes[0]].reshape(shape) * sim.dt
    output_spikes_pos = sim.data[left_probes[1]].reshape(shape) * sim.dt
    gabor_output = sim.data[left_probes[2]]

    # Save the output spikes as .npy files
    np.save('/home/avi/projects/nengox/data/model/sim_t.npy', sim_t)
    np.save('/home/avi/projects/nengox/data/model/output_spikes_neg.npy', output_spikes_neg)
    np.save('/home/avi/projects/nengox/data/model/output_spikes_pos.npy', output_spikes_pos)
    np.save('/home/avi/projects/nengox/data/model/gabor_output.npy', gabor_output)

    print("Saved output spikes and Gabor output to .npy files.")
