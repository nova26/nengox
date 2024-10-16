import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

def load_and_create_mp4(neg_file, pos_file, sim_t_file, shape, dt_frame, output_file):
    """
    Load spike data from .npy files and create an MP4 animation.

    Parameters:
    - neg_file: Path to the .npy file for negative spikes.
    - pos_file: Path to the .npy file for positive spikes.
    - sim_t_file: Path to the .npy file containing the simulation time array.
    - shape: Tuple of (time steps, height, width) for reshaping the data.
    - dt_frame: Time step for each frame in the animation.
    - output_file: Path to save the resulting MP4 file.
    """
    # Load the negative and positive spikes from .npy files
    output_spikes_neg = np.load(neg_file)
    output_spikes_pos = np.load(pos_file)
    sim_t = np.load(sim_t_file)

    print(f"Loaded negative spikes from {neg_file} with shape {output_spikes_neg.shape}")
    print(f"Loaded positive spikes from {pos_file} with shape {output_spikes_pos.shape}")
    print(f"Loaded simulation time array from {sim_t_file} with shape {sim_t.shape}")

    # Prepare for creating the animation
    fig = plt.figure()
    imgs = []
    t_frames = dt_frame * np.arange(int(round(sim_t[-1] / dt_frame)))

    for t_frame in t_frames:
        t0 = t_frame
        t1 = t_frame + dt_frame
        m = (sim_t >= t0) & (sim_t < t1)

        # Combine negative and positive spikes for visualization
        frame_img = np.zeros(shape[1:])
        frame_img -= output_spikes_neg[m].sum(axis=0)
        frame_img += output_spikes_pos[m].sum(axis=0)

        # Normalize the frame image for better visualization
        # if np.abs(frame_img).max() != 0:
        #     frame_img = frame_img / (np.abs(frame_img).max() + 1e-6)

        img = plt.imshow(frame_img, vmin=-1, vmax=1, animated=True)
        imgs.append([img])

    # Create and save the animation
    ani = ArtistAnimation(fig, imgs, interval=50, blit=True)
    ani.save(output_file, writer='ffmpeg')

    print(f"Saved animation to {output_file}")


# Parameters
neg_file = '/home/avi/projects/nengox/data/model/output_spikes_neg.npy'
pos_file = '/home/avi/projects/nengox/data/model/output_spikes_pos.npy'
sim_t_file = '/home/avi/projects/nengox/data/model/sim_t.npy'
shape = (len(np.load(sim_t_file)), 320, 240)  # Adjust based on your simulation's output
dt_frame = 0.01
output_file = '/home/avi/projects/nengox/data/model/loaded_spikes_animation.mp4'

# Call the function with the loaded data
load_and_create_mp4(neg_file, pos_file, sim_t_file, shape, dt_frame, output_file)

