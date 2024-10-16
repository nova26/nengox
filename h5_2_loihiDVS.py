
import numpy as np
from utils.eventslicer import EventSlicer

import tables as tb
import h5py
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import nengo_loihi


def create_vid(events_file_name, dvs_height, dvs_width):
    import matplotlib.pyplot as plt
    from matplotlib.animation import ArtistAnimation
    import numpy as np
    from tqdm import tqdm
    import nengo_loihi

    print("-I- starting to create video")

    # Load the events from the specified file
    dvs_events = nengo_loihi.dvs.DVSEvents.from_file(events_file_name)

    # Extract event timestamps
    t = dvs_events.events[:]["t"]
    t_length_us = t[-1] - t[0]

    # Set frame duration
    dt_frame_us = 20000
    t_frames = dt_frame_us * np.arange(int(round(t_length_us / dt_frame_us)))

    # Prepare figure and list of frames for the animation
    fig = plt.figure()
    imgs = []

    event_count = 0
    for t_frame in tqdm(t_frames):
        t0_us = t_frame
        t1_us = t_frame + dt_frame_us

        # Select events within the current frame time range
        m = (t >= t0_us) & (t < t1_us)
        events_m = dvs_events.events[m]

        event_count += np.sum(m)

        # Convert polarities to -1 and +1
        events_sign = 2.0 * events_m["p"] - 1

        # Ensure x and y coordinates are within the bounds (0, dvs_width) and (0, dvs_height)
        x = np.clip(events_m["x"], 0, dvs_width - 1)
        y = np.clip(events_m["y"], 0, dvs_height - 1)

        # Create a frame image of the current events
        frame_img = np.zeros((dvs_height, dvs_width))
        frame_img[y, x] = events_sign

        # Add the image to the list of frames
        img = plt.imshow(frame_img, vmin=-1, vmax=1, animated=True)
        imgs.append([img])

    # Clean up the DVS events
    del dvs_events
    print(f"-I- converted total of {event_count} events and {len(imgs)} images")

    # Create animation
    ani = ArtistAnimation(fig, imgs, interval=50, blit=True)

    # Save the animation as an MP4 file
    nv = events_file_name.split("/")[-1].split(".")[0]
    output_file = f'/home/avi/projects/nengox/data/vid/{nv}.mp4'
    ani.save(output_file, writer='ffmpeg')  # Or use 'imagemagick' for .gif

    print(f"Animation saved as {output_file}")


def process_file(f, width, height):
    event_filepath = Path(f)

    # Read using tables
    with tb.open_file(str(event_filepath), mode='r') as file:
        p_data = file.root.events.p[:100]  # Adjust for chunking
        df = pd.DataFrame({'p': p_data})
        print(df)

    # Read using h5py
    h5f = h5py.File(str(event_filepath), 'r')
    event_slicer = EventSlicer(h5f)

    # Get the events from the event slicer
    e = event_slicer.get_events(event_slicer.get_start_time_us(), event_slicer.get_final_time_us())

    eventsList = []
    p = e['p']
    x = e['x']
    y = e['y']
    t = e['t']
    t = t - t[0]  # Adjust time so that it starts from 0

    min_y = 50  # Example minimum value for y
    max_y = 100  # Example maximum value for y

    # Ensure x and y coordinates are within the bounds (0, width) and (0, height)
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, min_y, max_y)

    eventsList.append((t, p, x, y))

    # Create DVSEvents object
    dvs_events = nengo_loihi.dvs.DVSEvents()
    nbEvents = sum(len(xx) for _, _, xx, _ in eventsList)
    dvs_events.init_events(n_events=nbEvents)

    print("Fill the DVSEvents object with events data")
    i = 0
    for tt, p, xx, yy in tqdm(eventsList):
        ee = dvs_events.events[i: i + len(xx)]
        ee["t"] = tt
        ee["p"] = p
        ee["x"] = xx
        ee["y"] = yy
        i += len(xx)

    # Generate file name for events
    side = f.split("/")[-2]
    name = f.split("/")[-4]

    events_file_name = f'/home/avi/projects/nengox/data/{name}_{side}.events'
    dvs_events.write_file(events_file_name)
    print("Wrote %r" % events_file_name)

    # Call to create video from events
    create_vid(events_file_name, 50, width)  # Adjust


def convert(height,width):

    data_root = "/home/avi/projects/data/train_events/thun_00_a"
    right_camera = f"{data_root}/events/right/events.h5"
    left_camera =  f"{data_root}/events/left/events.h5"

    files = [right_camera, left_camera]

    for f in files:
        event_filepath = Path(f)
        h5f = h5py.File(str(event_filepath), 'r')
        event_slicer = EventSlicer(h5f)
        e = event_slicer.get_events(event_slicer.get_start_time_us(), event_slicer.get_final_time_us())

        eventsList = []
        p = e['p']
        x = e['x']
        y = e['y']
        t = e['t']
        t = t - t[0]  # Adjust time so that it starts from 0

        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        eventsList.append((t, p, x, y))

        # Create DVSEvents object
        dvs_events = nengo_loihi.dvs.DVSEvents()
        nbEvents = sum(len(xx) for tt, pp, xx, yy in eventsList)
        dvs_events.init_events(n_events=nbEvents)

        print("Fill the DVSEvents object with events data")
        i = 0
        for tt, p, xx, yy in tqdm(eventsList):
            ee = dvs_events.events[i: i + len(xx)]
            ee["t"] = tt
            ee["p"] = p
            ee["x"] = xx
            ee["y"] = yy
            i += len(xx)

        # Generate file name for events
        side = f.split("/")[-2]
        name = f.split("/")[-4]

        events_file_name = f'/home/avi/projects/nengox/data/{name}_{side}.events'
        dvs_events.write_file(events_file_name)
        print("Wrote %r" % events_file_name)

        # Call to create video from events
        create_vid(events_file_name, height, width)  # Adjust



if __name__ == '__main__':

    convert(100,200)

    # events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\syntactic_left.events'
    # create_syntactic_data(events_file_name,0)
    # create_vid(events_file_name,20,50)
    #
    #
    # events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\syntactic_right.events'
    # create_syntactic_data(events_file_name,20)
    # create_vid(events_file_name,20,50)
    #
    #
    #
