class LearningMetrics:
    def __init__(self, tracking_data):
        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        self.metrics = {}
        self.trials_data = None
        self.trial_durations = None

        # Detect trials and compute metrics
        self.detect_trials()
        self.compute_metrics()

    def detect_trials(self):
        """
        Detect individual trials based on negative peaks in the ball position derivative.
        """
        # Get ball data
        ball_data = self.tracking_data.balltrack.objects[0].dataset.copy()

        # Compute the derivative of the ball position using the standard column
        ball_data["position_derivative"] = ball_data["euclidean_distance"].diff()

        # Find negative peaks (indicating ball reset)
        peaks, properties = find_peaks(
            -ball_data["position_derivative"],
            height=self.fly.config.trial_peak_height,
            distance=self.fly.config.trial_peak_distance,
        )

        # Assign trial numbers
        ball_data["trial"] = 0
        trial_number = 1
        previous_peak = 0

        for peak in peaks:
            ball_data.iloc[
                previous_peak : peak + 1, ball_data.columns.get_loc("trial")
            ] = trial_number
            trial_number += 1
            previous_peak = peak + 1

        # Assign last trial
        ball_data.iloc[previous_peak:, ball_data.columns.get_loc("trial")] = (
            trial_number
        )

        # Add trial_frame and trial_time columns
        ball_data["trial_frame"] = ball_data.groupby("trial").cumcount()
        ball_data["trial_time"] = ball_data["trial_frame"] / self.fly.experiment.fps

        # Store the data with trial information
        self.trials_data = ball_data

        # Clean trials (skip initial frames, trim at max position)
        self.clean_trials()

        # Compute trial durations
        self.compute_trial_durations()

        return self.trials_data

    def clean_trials(self):
        """
        Clean trial data by removing initial frames and trimming at maximum position.
        """
        cleaned_data = []

        for trial in self.trials_data["trial"].unique():
            # Get data for this trial
            trial_data = self.trials_data[self.trials_data["trial"] == trial].copy()

            # Skip initial frames
            if len(trial_data) > self.fly.config.trial_skip_frames:
                trial_data = trial_data.iloc[self.fly.config.trial_skip_frames :]

            # Find max position and trim
            max_index = trial_data["euclidean_distance"].idxmax()
            trial_data = trial_data.loc[:max_index]

            cleaned_data.append(trial_data)

        if cleaned_data:
            self.trials_data = pd.concat(cleaned_data).reset_index(drop=True)

    def compute_trial_durations(self):
        """
        Compute the duration of each trial.
        """
        durations = []

        for trial in self.trials_data["trial"].unique():
            trial_data = self.trials_data[self.trials_data["trial"] == trial]
            duration = trial_data["time"].max() - trial_data["time"].min()
            durations.append({"trial": trial, "duration": duration})

        self.trial_durations = pd.DataFrame(durations)

    def compute_metrics(self):
        """
        Compute metrics for learning experiments.
        """
        # Basic metrics about trials
        self.metrics["num_trials"] = len(self.trials_data["trial"].unique())

        if not self.trial_durations.empty:
            self.metrics["mean_trial_duration"] = self.trial_durations[
                "duration"
            ].mean()
            self.metrics["trial_durations"] = self.trial_durations["duration"].tolist()
            self.metrics["trial_numbers"] = self.trial_durations["trial"].tolist()

        # Map interaction events to trials
        self.map_interactions_to_trials()

    def map_interactions_to_trials(self):
        """
        Associate interaction events with trials.
        """
        if not hasattr(self.tracking_data, "interaction_events"):
            return

        # Create a mapping of frame to trial
        frame_to_trial = dict(zip(self.trials_data.index, self.trials_data["trial"]))

        # Map each interaction event to a trial
        trial_interactions = {}

        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                for event in events:
                    start_frame, _ = event[0], event[1]

                    # Find trial for the event (use start frame)
                    if start_frame in frame_to_trial:
                        trial = frame_to_trial[start_frame]

                        if trial not in trial_interactions:
                            trial_interactions[trial] = []

                        trial_interactions[trial].append(event)

        self.metrics["trial_interactions"] = trial_interactions

        # Count interactions per trial
        self.metrics["interactions_per_trial"] = {
            trial: len(events) for trial, events in trial_interactions.items()
        }