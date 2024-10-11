import numpy as np

from utils.eventslicer import EventSlicer

rng = np.random.RandomState(0)
import nengo_loihi
nengo_loihi.set_defaults()

import nengo
import nengo_loihi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import h5py
import tables as tb
import pandas as pd
from matplotlib.animation import FFMpegWriter


class DVSNode:
    def __init__(self, event_slicer: EventSlicer, dt: float):
        self.event_slicer = event_slicer
        self.dt = dt
        self.sim_time = 0.0
        e = event_slicer.get_events(event_slicer.get_start_time_us(), event_slicer.get_start_time_us()+1000)
        self.offset = e['t'][0]

    def __call__(self, t):
        # Time window for this timestep (in microseconds)
        t_start_us = int(self.sim_time * 1e6)
        t_end_us = int((self.sim_time + self.dt) * 1e6)

        # Get events within the time window
        events = self.event_slicer.get_events(t_start_us, t_end_us)

        # Initialize outputs as zero arrays if no events are found
        if events is None:
            return np.zeros(self.event_slicer.events['x'].size * 2)  # No events in this time step

        # Separate positive and negative polarity events
        pos_events = events['p'] == 1
        neg_events = events['p'] == 0

        # Extract event values (e.g., x and y coordinates)
        pos_event_values = np.stack([events['x'][pos_events], events['y'][pos_events]], axis=-1)
        neg_event_values = np.stack([events['x'][neg_events], events['y'][neg_events]], axis=-1)

        # Pad the shorter output with zeros to match dimensions
        max_len = max(len(pos_event_values), len(neg_event_values))
        if len(pos_event_values) < max_len:
            pos_event_values = np.pad(pos_event_values, ((0, max_len - len(pos_event_values)), (0, 0)))
        if len(neg_event_values) < max_len:
            neg_event_values = np.pad(neg_event_values, ((0, max_len - len(neg_event_values)), (0, 0)))

        # Flatten the arrays and concatenate them: positive events first, negative second
        output = np.concatenate([pos_event_values.flatten(), neg_event_values.flatten()])

        # Update simulation time
        self.sim_time += self.dt

        return output

    def get_next_event_window(self):
        """Retrieve the next window of events based on dt."""
        # Time window for this timestep (in microseconds)
        t_start_us = int(self.sim_time * 1e6)
        t_end_us = int((self.sim_time + self.dt) * 1e6)

        # Get events within the time window
        events = self.event_slicer.get_events(t_start_us, t_end_us)

        # Update simulation time for the next call
        self.sim_time += self.dt

        # Return events if available
        return events if events is not None else None

def run_simulation(event_slicer: EventSlicer, duration: float = 5.0, dt: float = 0.001, window_ms: float = 30.0):
    dvs_node = DVSNode(event_slicer, dt=dt)
    frames = []
    sim_time = 0.0
    window_us = window_ms * 1e3  # Convert window to microseconds
    num_windows = int(duration * 1e3 // window_ms)  # Number of windows based on the time window size

    # Convert 'x' and 'y' datasets to NumPy arrays and get the max values
    max_x = np.max(np.array(event_slicer.events['x']))
    max_y = np.max(np.array(event_slicer.events['y']))

    for i in range(num_windows):
        # Aggregate events over 30ms window
        t_start = i * window_us / 1e6
        t_end = (i + 1) * window_us / 1e6

        # Initialize empty images for this window based on max x and y
        pos_img = np.zeros((max_y + 1, max_x + 1))
        neg_img = np.zeros_like(pos_img)

        # Collect events over dt for the entire 30ms window
        while sim_time < t_end:
            events = dvs_node.get_next_event_window()
            if events:
                pos_events = events['p'] == 1
                neg_events = events['p'] == 0

                # Update images based on the x, y positions of positive and negative events
                pos_img[events['y'][pos_events], events['x'][pos_events]] += 1
                neg_img[events['y'][neg_events], events['x'][neg_events]] += 1

            sim_time += dt

        # Combine positive and negative polarity events into a single image
        combined_img = np.zeros_like(pos_img)
        combined_img += pos_img  # Positive events brighten the image
        combined_img -= neg_img  # Negative events darken the image

        frames.append(combined_img)

    return frames

# Function to save the frames as an ffmpeg video
def save_to_video(frames, output_filename='output.mp4', fps=30):
    fig, ax = plt.subplots()

    # Use FFmpeg writer to write the video file
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_filename, dpi=100):
        for frame in frames:
            ax.clear()
            ax.imshow(frame, cmap='gray', vmin=-1, vmax=1)
            writer.grab_frame()

    print(f"Video saved to {output_filename}")


def main():

    event_filepath = r'C:\Users\USER\Downloads\MyPHDWork\train_events\thun_00_a\events\right\events.h5'

    with tb.open_file(str(event_filepath), mode='r') as file:
        p_data = file.root.events.p[:100]  # Adjust for chunking
        df = pd.DataFrame({'p': p_data})
        print(df)


    h5f = h5py.File(str(event_filepath), 'r')
    event_slicer = EventSlicer(h5f)


    frames = run_simulation(event_slicer, duration=5.0, dt=0.001, window_ms=30.0)

    # Save the frames to a video
    save_to_video(frames, output_filename='dvs_events.mp4', fps=30)


if __name__ == '__main__':
    main()
