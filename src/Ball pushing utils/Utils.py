def save_object(obj, filename):
    """Save a custom object as a pickle file.

    Args:
        obj (object): The object to be saved.
        filename (Path): The path where to save the object. No need to add the .pkl extension.
    """
    # Ensure filename is a Path object
    filename = Path(filename)

    # If the filename does not end with .pkl, add it
    if filename.suffix != ".pkl":
        filename = filename.with_suffix(".pkl")

    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Load a custom object from a pickle file.

    Args:
        filename (Pathlib path): the path to the object. No need to add the .pkl extension.
    """

    # Ensure filename is a Path object
    filename = Path(filename)

    # If the filename does not end with .pkl, add it
    if filename.suffix != ".pkl":
        filename = filename.with_suffix(".pkl")

    with open(filename, "rb") as input:
        obj = pickle.load(input)

    return obj


def find_interaction_events(
    protagonist1,
    protagonist2,
    nodes1=["Lfront", "Rfront"],
    nodes2=["x_centre", "y_centre"],
    threshold=[0, 11],
    gap_between_events=4,
    event_min_length=2,
    fps=29,
):
    """
    This function finds interaction events where the specified nodes of two protagonists are within a certain distance for a minimum amount of time.

    Parameters:
    protagonist1 (DataFrame): DataFrame containing the first protagonist's tracking data.
    protagonist2 (DataFrame): DataFrame containing the second protagonist's tracking data.
    nodes1 (list): List of nodes for the first protagonist to check the distance with the second protagonist (e.g., ["Lfront", "Rfront"]).
    nodes2 (list): List of nodes for the second protagonist to check the distance with the first protagonist (e.g., ["Centre",]).
    threshold (int): The distance threshold (in pixels) for the nodes to be considered in interaction.
    gap_between_events (int): The minimum gap required between two events, expressed in seconds.
    event_min_length (int): The minimum length of an event, expressed in seconds.
    fps (int): Frames per second of the video.

    Returns:
    list: A list of interaction events, where each event is represented as [start_frame, end_frame, duration].
    """

    # Convert the gap between events and the minimum event length from seconds to frames
    gap_between_events = gap_between_events * fps
    event_min_length = event_min_length * fps

    # Initialize a list to store distances for all specified nodes
    distances = []

    # Compute the Euclidean distance for each specified node
    for node1 in nodes1:
        for node2 in nodes2:

            # x1 = protagonist1[f"x_{node1}"]
            # y1 = protagonist1[f"y_{node1}"]
            # x2 = protagonist2[f"x_{node2}"]
            # y2 = protagonist2[f"y_{node2}"]

            # # Check for NaN values
            # if x1.isna().any() or y1.isna().any() or x2.isna().any() or y2.isna().any():
            #     print(f"NaN values found in data for nodes {node1} and {node2}")
            #     print(f"x1: {x1}")
            #     print(f"y1: {y1}")
            #     print(f"x2: {x2}")
            #     print(f"y2: {y2}")

            # # Print intermediate values
            # print(f"x1: {x1}")
            # print(f"y1: {y1}")
            # print(f"x2: {x2}")
            # print(f"y2: {y2}")

            distances_node = np.sqrt(
                (protagonist1[f"x_{node1}"] - protagonist2[f"x_{node2}"]) ** 2
                + (protagonist1[f"y_{node1}"] - protagonist2[f"y_{node2}"]) ** 2
            )
            # print(f"Distances for {node1} and {node2}: {distances_node}")
            distances.append(distances_node)

    # Combine distances to find frames where any node is within the threshold distance
    combined_distances = np.min(distances, axis=0)
    interaction_frames = np.where(
        (np.array(combined_distances) > threshold[0])
        & (np.array(combined_distances) < threshold[1])
    )[0]

    # Debug: Print the combined distances and interaction frames
    # print(f"Combined distances: {combined_distances}")
    # print(f"Interaction frames: {interaction_frames}")

    # If no interaction frames are found, return an empty list
    if len(interaction_frames) == 0:
        return []

    # Find the distance between consecutive interaction frames
    distance_betw_frames = np.diff(interaction_frames)

    # Find the points where the distance between frames is greater than the gap between events
    split_points = np.where(distance_betw_frames > gap_between_events)[0]

    # Add the first and last points to the split points
    split_points = np.insert(split_points, 0, -1)
    split_points = np.append(split_points, len(interaction_frames) - 1)

    # Initialize the list of interaction events
    interaction_events = []

    # Iterate over the split points to find events
    for f in range(0, len(split_points) - 1):
        # Define the start and end of the region of interest (ROI)
        start_roi = interaction_frames[split_points[f] + 1]
        end_roi = interaction_frames[split_points[f + 1]]

        # Calculate the duration of the event
        duration = end_roi - start_roi

        # If the duration of the event is greater than the minimum length, add the event to the list
        if duration > event_min_length:
            interaction_events.append([start_roi, end_roi, duration])

    return interaction_events


def find_interaction_start(
    data,
    distance_col,
    frame_col,
    threshold_multiplier=1.5,
    window_size=30,
    min_plateau_length=50,
    peak_prominence=1,
    peak_window_size=10,
):
    # Work on a copy to avoid modifying original data
    data = data.copy()

    # Smoothing and derivative calculation
    data["smoothed_distance"] = savgol_filter(
        data[distance_col], window_length=11, polyorder=3
    )
    data["smoothed_diff"] = data["smoothed_distance"].diff()
    data["smoothed_accel"] = data["smoothed_diff"].diff()  # Second derivative

    # Dynamic threshold for plateaus based on rolling standard deviation
    rolling_std = data["smoothed_diff"].rolling(window=window_size).std()
    dynamic_threshold = (
        rolling_std.mean() * threshold_multiplier
    )  # Adjust multiplier as needed

    # Plateau detection with dynamic threshold
    plateau_mask = data["smoothed_diff"].abs() < dynamic_threshold
    plateau_groups = (plateau_mask != plateau_mask.shift()).cumsum()

    # Initialize plateau markers
    data["plateau_start"] = 0
    valid_plateaus = (
        data[plateau_mask]
        .groupby(plateau_groups)
        .filter(lambda x: len(x) >= min_plateau_length)
    )
    if not valid_plateaus.empty:
        start_indices = valid_plateaus.groupby(plateau_groups).head(1).index
        data.loc[start_indices, "plateau_start"] = 1

    # Peak detection with stricter prominence
    peaks, _ = find_peaks(
        -data["smoothed_distance"], prominence=peak_prominence, width=3
    )

    # Refine peak detection for better alignment
    refined_peaks = []
    for peak in peaks:
        if peak > 0 and peak < len(data) - 1:
            # Perform local search around the detected peak to refine position
            local_region = data.iloc[
                max(0, peak - peak_window_size) : min(
                    len(data), peak + peak_window_size
                )
            ]
            true_peak_idx = local_region[
                distance_col
            ].idxmin()  # Find true minimum in this region
            refined_peaks.append(true_peak_idx)

    # Combine plateau and refined peak detections
    plateau_indices = data[data["plateau_start"] == 1].index
    all_candidates = sorted(list(plateau_indices) + refined_peaks)

    # Fallback to minimum distance if no markers found
    if not all_candidates:
        return data[distance_col].idxmin()

    return all_candidates[0]


def filter_experiments(source, **criteria):
    """Generates a list of Experiment objects based on criteria.

    Args:
        source (list): A list of flies, experiments or folders to create Experiment objects from.
        criteria (dict): A dictionary of criteria to filter the experiments.

    Returns:
        list: A list of Experiment objects that match the criteria.
    """

    flies = []

    # If the source is a list of flies, check directly for the criteria in flies
    if isinstance(source[0], Fly):
        for fly in source:
            if all(
                fly.arena_metadata.get(key) == value for key, value in criteria.items()
            ):
                flies.append(fly)

    else:
        if isinstance(source[0], Experiment):
            Exps = source

        else:
            # Create Experiment objects from the folders
            Exps = [Experiment(f) for f in source]

        # Get a list of flies based on the criteria
        for exp in Exps:
            for fly in exp.flies:
                if all(
                    fly.arena_metadata.get(key) == value
                    for key, value in criteria.items()
                ):
                    flies.append(fly)

    return flies


def load_fly(
    mp4_file,
    experiment,
    # experiment_type,
):
    print(f"Loading fly from {mp4_file.parent}")
    try:
        fly = Fly(
            mp4_file.parent,
            experiment=experiment,
            # experiment_type=experiment_type,
        )
        if fly.tracking_data and fly.tracking_data.valid_data:
            return fly
    except TypeError as e:
        print(f"Error while loading fly from {mp4_file.parent}: {e}")
    return None