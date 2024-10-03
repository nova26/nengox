import nengo
import nengo_loihi

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import ArtistAnimation

nengo_loihi.set_defaults()

rng = np.random.RandomState(0)

right_camera = r'C:\Users\USER\Downloads\MyPHDWork\train_events\thun_00_a\events\right\events.h5'
left_camera = r'C:\Users\USER\Downloads\MyPHDWork\train_events\thun_00_a\events\left\events.h5'

dvs_height = 480
dvs_width = 640

# the length of time to generate data for, in seconds and in microseconds
t_length = 1.0
t_length_us = int(1e6 * t_length)

dvs_events = nengo_loihi.dvs.DVSEvents.from_file(left_camera)

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
