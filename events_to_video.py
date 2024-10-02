import argparse
from pathlib import Path

import numpy as np
import skvideo.io
from tqdm import tqdm
import pandas as pd
from visualization.eventreader import EventReader

import h5py
def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    img[mask==0]=[255,255,255]

#    img[mask==-1]=[255,0,0]
    img[mask==-1]=[0,0,255]

    img[mask==1]=[0,0,255]
    return img

if __name__ == '__main__':
    import tables as tb

    event_filepath = Path(r'C:\Users\USER\Downloads\train_events\zurich_city_00_a\events\left\events.h5')
    video_filepath = Path(r'C:\Users\USER\Downloads\output_video.mp4')  # Ensure a valid extension like .mp4
    dt = 10
    height = 480
    width = 640

    with tb.open_file(event_filepath, mode='r') as file:
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

    assert video_filepath.parent.is_dir(), "Directory {} does not exist".format(str(video_filepath.parent))

    writer = skvideo.io.FFmpegWriter(str(video_filepath))  # Convert Path to str
    for events in tqdm(EventReader(event_filepath, dt)):
        p = events['p']
        x = events['x']
        y = events['y']
        t = events['t']
        img = render(x, y, p, height, width)
        writer.writeFrame(img)
    writer.close()