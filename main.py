import tables as tb
import pandas as pd

file_path = r'C:\Users\USER\Downloads\train_events\zurich_city_00_a\events\left\events.h5'

with tb.open_file(file_path, mode='r') as file:
    # Access the events group and datasets
    p_data = file.root.events.p[:100]  # Adjust for chunking
    t_data = file.root.events.t[:100]
    x_data = file.root.events.x[:100]
    y_data = file.root.events.y[:100]

    df = pd.DataFrame({'p': p_data, 't': t_data, 'x': x_data, 'y': y_data})
    print(df)
