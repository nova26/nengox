import argparse
from pathlib import Path
import nengo
import nengo_loihi
import numpy as np
import skvideo.io
from tqdm import tqdm
import pandas as pd
from visualization.eventreader import EventReader

import h5py


def convert():
    import tables as tb

    right_camera = r'C:\Users\USER\Downloads\MyPHDWork\train_events\thun_00_a\events\right\events.h5'
    left_camera = r'C:\Users\USER\Downloads\MyPHDWork\train_events\thun_00_a\events\left\events.h5'

    event_filepath = Path(right_camera)

    with tb.open_file(str(event_filepath), mode='r') as file:
        # Access the events group and datasets
        p_data = file.root.events.p[:100]  # Adjust for chunking
        t_data = file.root.events.t[:100]
        x_data = file.root.events.x[:100]
        y_data = file.root.events.y[:100]

        df = pd.DataFrame({'p': p_data, 't': t_data, 'x': x_data, 'y': y_data})
        print(df)

    with h5py.File(event_filepath, 'r') as h5f:
        for key in h5f.keys():
            print(f"Key: {key}, Dataset: {h5f[key]}")


    dtMilli = 1000
    height = 480
    width = 640

    events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\dvs-from-file-events.events'
    eventsList = []

    for events in tqdm(EventReader(event_filepath, dtMilli)):
        p = events['p']
        x = events['x']
        y = events['y']
        t = events['t']
        eventsList.append((t, p, x, y))


    dvs_events = nengo_loihi.dvs.DVSEvents()
    nbEvents = sum(len(xx) for _, _, xx, _ in eventsList)
    dvs_events.init_events(n_events=nbEvents)

    i = 0
    for tt, p, xx, yy in eventsList:
        ee = dvs_events.events[i: i + len(xx)]
        ee["t"] = tt
        ee["p"] = p
        ee["x"] = xx
        ee["y"] = yy
        i += len(xx)

    dvs_events.write_file(events_file_name)
    print("Wrote %r" % events_file_name)


def display():
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from matplotlib.animation import ArtistAnimation

    t_length = 1.0
    t_length_us = int(1e6 * t_length)

    dvs_height = 480
    dvs_width = 640

    events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\dvs-from-file-events.events'
    dvs_events = nengo_loihi.dvs.DVSEvents.from_file(events_file_name)

    dt_frame_us = 20e3
    t_frames = dt_frame_us * np.arange(int(round(t_length_us / dt_frame_us)))

    fig = plt.figure()
    imgs = []
    for t_frame in t_frames:
        t0_us = t_frame
        t1_us = t_frame + dt_frame_us
        t = dvs_events.events[:]["t"]
        m = (t >= t0_us) & (t < t1_us)
        events_m = dvs_events.events[m]

        # show "off" (0) events as -1 and "on" (1) events as +1
        events_sign = 2.0 * events_m["p"] - 1

        frame_img = np.zeros((dvs_height, dvs_width))
        frame_img[events_m["y"], events_m["x"]] = events_sign

        img = plt.imshow(frame_img, vmin=-1, vmax=1, animated=True)
        imgs.append([img])

    del dvs_events

    ani = ArtistAnimation(fig, imgs, interval=50, blit=True)
    HTML(ani.to_html5_video())

if __name__ == '__main__':
    display()
