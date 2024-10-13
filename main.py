import os
import numpy as np
from nengo_extras.vision import Gabor
import nengo
import nengo_loihi
from dvs_utils import create_input_ensembles, create_filter_ensembles, gabor_transform, gaussian_transform_matrix, \
    visualize_and_save

nengo_loihi.set_defaults()
rng = np.random.RandomState(0)

t_length = 2
dt_frame = 0.01
left_events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\thun_00_a_left.events'
right_events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\thun_00_a_right.events'


if __name__ == '__main__':
    model = nengo.Network(label="NEF summary")

    with model:
        print("Creating DVS object")
        left_dvs_process = nengo_loihi.dvs.DVSFileChipProcess(
            file_path=left_events_file_name, channels_last=True, dvs_height=240, dvs_width=320,
        )
        u_left = nengo.Node(left_dvs_process)

        print("Creating input ensembles to DVS")
        left_positive_ensemble, left_negative_ensemble = create_input_ensembles(left_dvs_process.height * left_dvs_process.width)
        print("Creating filter ensembles")
        left_positive_gabor_ensemble, left_positive_gaussian_ensemble = create_filter_ensembles(left_dvs_process.height * left_dvs_process.width)
        left_negative_gabor_ensemble,left_negative_gaussian_ensemble = create_filter_ensembles(left_dvs_process.height * left_dvs_process.width)

        print("Connecting input ensembles to DVS")
        nengo.Connection(u_left[1:: left_dvs_process.polarity], left_positive_ensemble.neurons, transform=1.0)
        nengo.Connection(u_left[0:: left_dvs_process.polarity], left_negative_ensemble.neurons, transform=1.0)

        print("Connecting Gabor filter to input ensembles")
        left_gabor_bank = Gabor().generate(1, (11, 11), rng)  # Single Gabor filter for convolution
        left_gabor_transform_matrix = gabor_transform(left_dvs_process.height, left_dvs_process.width, left_gabor_bank)
        nengo.Connection(left_positive_ensemble.neurons, left_positive_gabor_ensemble.neurons, transform=left_gabor_transform_matrix)
        nengo.Connection(left_negative_ensemble.neurons, left_negative_gabor_ensemble.neurons, transform=left_gabor_transform_matrix)

        print("Connecting Gaussian filter to input ensembles")
        left_gaussian_transform_matrix_values = gaussian_transform_matrix(left_dvs_process.height, left_dvs_process.width, sigma=1.0)
        nengo.Connection(left_positive_ensemble.neurons, left_positive_gaussian_ensemble.neurons,
                         transform=left_gaussian_transform_matrix_values)
        nengo.Connection(left_negative_ensemble.neurons, left_negative_gaussian_ensemble.neurons,
                         transform=left_gaussian_transform_matrix_values)

        left_probes = [nengo.Probe(left_negative_ensemble.neurons),
                  nengo.Probe(left_positive_ensemble.neurons),
                  nengo.Probe(left_positive_gabor_ensemble.neurons),
                  nengo.Probe(left_positive_gaussian_ensemble.neurons),
                  nengo.Probe(left_negative_gabor_ensemble.neurons),
                  nengo.Probe(left_negative_gaussian_ensemble.neurons)
                  ]
        ######################################################

        right_dvs_process = nengo_loihi.dvs.DVSFileChipProcess(
            file_path=right_events_file_name, channels_last=True, dvs_height=240, dvs_width=320,
        )
        u_right = nengo.Node(right_dvs_process)

        print("Creating input ensembles to DVS")
        right_positive_ensemble, right_negative_ensemble = create_input_ensembles(
            right_dvs_process.height * right_dvs_process.width)
        print("Creating filter ensembles")
        right_positive_gabor_ensemble, right_positive_gaussian_ensemble = create_filter_ensembles(
            right_dvs_process.height * right_dvs_process.width)
        right_negative_gabor_ensemble, right_negative_gaussian_ensemble = create_filter_ensembles(
            right_dvs_process.height * right_dvs_process.width)

        print("Connecting input ensembles to DVS")
        nengo.Connection(u_right[1:: right_dvs_process.polarity], right_positive_ensemble.neurons, transform=1.0)
        nengo.Connection(u_right[0:: right_dvs_process.polarity], right_negative_ensemble.neurons, transform=1.0)

        print("Connecting Gabor filter to input ensembles")
        right_gabor_bank = Gabor().generate(1, (11, 11), rng)  # Single Gabor filter for convolution
        right_gabor_transform_matrix = gabor_transform(right_dvs_process.height, right_dvs_process.width,
                                                       right_gabor_bank)
        nengo.Connection(right_positive_ensemble.neurons, right_positive_gabor_ensemble.neurons,
                         transform=left_gabor_transform_matrix)
        nengo.Connection(right_negative_ensemble.neurons, right_negative_gabor_ensemble.neurons,
                         transform=left_gabor_transform_matrix)

        print("Connecting Gaussian filter to input ensembles")
        right_gaussian_transform_matrix_values = gaussian_transform_matrix(right_dvs_process.height,
                                                                           right_dvs_process.width,
                                                                           sigma=1.0)
        nengo.Connection(right_positive_ensemble.neurons, right_positive_gaussian_ensemble.neurons,
                         transform=right_gaussian_transform_matrix_values)
        nengo.Connection(right_negative_ensemble.neurons, right_negative_gaussian_ensemble.neurons,
                         transform=right_gaussian_transform_matrix_values)

        right_probes = [nengo.Probe(right_negative_ensemble.neurons),
                        nengo.Probe(right_positive_ensemble.neurons),
                        nengo.Probe(right_positive_gabor_ensemble.neurons),
                        nengo.Probe(right_positive_gaussian_ensemble.neurons),
                        nengo.Probe(right_negative_gabor_ensemble.neurons),
                        nengo.Probe(right_negative_gaussian_ensemble.neurons)
                        ]

    print("Running simulation")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(t_length)

    sim_t = sim.trange()
    shape = (len(sim_t), left_dvs_process.height, left_dvs_process.width)
    left_output_spikes_neg = sim.data[left_probes[0]].reshape(shape) * sim.dt
    left_output_spikes_pos = sim.data[left_probes[1]].reshape(shape) * sim.dt
    left_gabor_filtered_output = sim.data[left_probes[2]].reshape(shape) * sim.dt
    left_gaussian_filtered_output = sim.data[left_probes[3]].reshape(shape) * sim.dt

    t_frames = dt_frame * np.arange(int(round(t_length / dt_frame)))

    # Visualize and save input ensemble output
    visualize_and_save(left_output_spikes_pos - left_output_spikes_neg, shape, sim_t, t_frames,
                       os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data\model_out', 'left_model_out_input.mp4'))

    # Visualize and save Gabor-filtered output
    visualize_and_save(left_gabor_filtered_output, shape, sim_t, t_frames,
                       os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data\model_out', 'left_gabor_filtered_model_out.mp4'))

    # Visualize and save Gaussian-filtered output
    visualize_and_save(left_gaussian_filtered_output, shape, sim_t, t_frames,
                       os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data\model_out', 'left_gaussian_filtered_model_out.mp4'))

    ###################################################
    right_output_spikes_neg = sim.data[right_probes[0]].reshape(shape) * sim.dt
    right_output_spikes_pos = sim.data[right_probes[1]].reshape(shape) * sim.dt
    right_gabor_filtered_output = sim.data[right_probes[2]].reshape(shape) * sim.dt
    right_gaussian_filtered_output = sim.data[right_probes[3]].reshape(shape) * sim.dt

    t_frames = dt_frame * np.arange(int(round(t_length / dt_frame)))

    # Visualize and save input ensemble output
    visualize_and_save(right_output_spikes_pos - right_output_spikes_neg, shape, sim_t, t_frames,
                       os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data\model_out', 'right_model_out_input.mp4'))

    # Visualize and save Gabor-filtered output
    visualize_and_save(right_gabor_filtered_output, shape, sim_t, t_frames,
                       os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data\model_out', 'right_gabor_filtered_model_out.mp4'))

    # Visualize and save Gaussian-filtered output
    visualize_and_save(right_gaussian_filtered_output, shape, sim_t, t_frames,
                       os.path.join(r'C:\Users\USER\PycharmProjects\nengox\data\model_out', 'right_gaussian_filtered_model_out.mp4'))
