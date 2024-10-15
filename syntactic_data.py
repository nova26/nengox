




def create_syntactic_data(destination_path, x_offset=0):
    import numpy as np
    from tqdm import tqdm
    import nengo_loihi
    from pathlib import Path

    # Image dimensions
    height = 20
    width = 50
    time_span_us = 11900998
    pulse_duration_us = 20000  # 20,000 microseconds for active events
    total_pulses = time_span_us // (pulse_duration_us * 2)  # Total number of on-off cycles

    # Synthetic event data
    eventsList = []

    # Generate plus sign coordinates
    plus_sign_width = 10
    plus_sign_height = 5
    polarity = 1  # Positive polarity for plus sign

    # Center the plus sign in the middle of the image
    center_x = width // 2
    center_y = height // 2

    # Create horizontal and vertical segments of the plus sign with the applied x_offset
    x_horiz = np.arange(center_x - plus_sign_width // 2, center_x + plus_sign_width // 2) + x_offset
    y_horiz = np.full_like(x_horiz, center_y)

    y_vert = np.arange(center_y - plus_sign_height // 2, center_y + plus_sign_height // 2)
    x_vert = np.full_like(y_vert, center_x) + x_offset

    # Combine the horizontal and vertical parts of the plus sign
    x = np.concatenate([x_horiz, x_vert])
    y = np.concatenate([y_horiz, y_vert])

    # Polarity array for the plus sign
    p = np.full_like(x, polarity)

    # Generate pulsed events: on for 20,000 us, off for 20,000 us, and so on
    for pulse in range(total_pulses):
        # "On" phase: generate plus sign events
        start_time = pulse * pulse_duration_us * 2  # Start time of the on phase
        end_time = start_time + pulse_duration_us  # End time of the on phase

        # Create time array for the on phase (20,000 microseconds)
        t_on = np.random.uniform(start_time, end_time, size=len(x))

        # Append the events for the "on" phase to the eventsList
        eventsList.append((t_on, p, x, y))

        # "Off" phase: no events are generated, so no need to append anything

    # Create DVSEvents object
    dvs_events = nengo_loihi.dvs.DVSEvents()
    nbEvents = sum(len(xx) for _, _, xx, _ in eventsList)
    dvs_events.init_events(n_events=nbEvents)

    # Fill the DVSEvents object with synthetic data
    i = 0
    for tt, pp, xx, yy in tqdm(eventsList):
        ee = dvs_events.events[i: i + len(xx)]
        ee["t"] = tt
        ee["p"] = pp
        ee["x"] = xx
        ee["y"] = yy
        i += len(xx)

    # Writing the synthetic event data to the destination path
    dvs_events.write_file(destination_path)
    print(f"Wrote {destination_path}")

