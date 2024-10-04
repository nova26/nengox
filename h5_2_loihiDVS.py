from pathlib import Path
import nengo_loihi
import numpy as np

import pandas as pd
from tqdm import tqdm

from utils.eventslicer import EventSlicer

import h5py


def convert():
    import tables as tb

    right_camera = r'C:\Users\USER\Downloads\MyPHDWork\train_events\thun_00_a\events\right\events.h5'
    left_camera = r'C:\Users\USER\Downloads\MyPHDWork\train_events\thun_00_a\events\left\events.h5'


    files = [right_camera,left_camera]

    for f in files:

        event_filepath = Path(f)

        with tb.open_file(str(event_filepath), mode='r') as file:
            p_data = file.root.events.p[:100]  # Adjust for chunking
            df = pd.DataFrame({'p': p_data})
            print(df)


        h5f = h5py.File(str(event_filepath), 'r')
        event_slicer = EventSlicer(h5f)

        e = event_slicer.get_events(event_slicer.get_start_time_us(),event_slicer.get_final_time_us())

        eventsList = []
        p = e['p']
        x = e['x']
        y = e['y']
        t = e['t']
        t = t - t[0]
        eventsList.append((t, p, x, y))

        dvs_events = nengo_loihi.dvs.DVSEvents()
        nbEvents = sum(len(xx) for _, _, xx, _ in eventsList)
        dvs_events.init_events(n_events=nbEvents)

        i = 0
        for tt, p, xx, yy in tqdm(eventsList):
            ee = dvs_events.events[i: i + len(xx)]
            ee["t"] = tt
            ee["p"] = p
            ee["x"] = xx
            ee["y"] = yy
            i += len(xx)

        side = f.split("\\")[-2]
        name =f.split("\\")[-4]

        events_file_name = rf'C:\Users\USER\PycharmProjects\nengox\data\{name}_{side}.events'
        dvs_events.write_file(events_file_name)
        print("Wrote %r" % events_file_name)


def create_vid():
    import matplotlib.pyplot as plt
    from matplotlib.animation import ArtistAnimation

    dvs_height = 480
    dvs_width = 640

    events_file_name = r'C:\Users\USER\PycharmProjects\nengox\data\thun_00_a_left.events'
    dvs_events = nengo_loihi.dvs.DVSEvents.from_file(events_file_name)

    t = dvs_events.events[:]["t"]
    t_length_us = t[-1] - t[0]

    dt_frame_us = 20e3
    t_frames = dt_frame_us * np.arange(int(round(t_length_us / dt_frame_us)))

    fig = plt.figure()
    imgs = []

    event_count = 0
    for t_frame in tqdm(t_frames):
        t0_us = t_frame
        t1_us = t_frame + dt_frame_us

        m = (t >= t0_us) & (t < t1_us)
        events_m = dvs_events.events[m]

        event_count+=np.sum(m)

        events_sign = 2.0 * events_m["p"] - 1

        frame_img = np.zeros((dvs_height, dvs_width))
        frame_img[events_m["y"], events_m["x"]] = events_sign

        img = plt.imshow(frame_img, vmin=-1, vmax=1, animated=True)
        imgs.append([img])

    del dvs_events
    print(f"-I- converted total of {event_count} events and {len(imgs)} images")

    ani = ArtistAnimation(fig, imgs, interval=50, blit=True)

    nv = events_file_name.split("\\")[-1].split(".")[0]
    ani.save(fr"C:\Users\USER\PycharmProjects\nengox\data\{nv}.mp4", writer='ffmpeg')  # Or use 'imagemagick' for .gif

    print("Animation saved as dvs_events_animation.mp4")

if __name__ == '__main__':
    # convert()
    create_vid()
