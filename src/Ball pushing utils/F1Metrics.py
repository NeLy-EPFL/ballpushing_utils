class F1Metrics:
    def __init__(self, tracking_data):
        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        self.metrics = {}
        self.compute_metrics()

    def compute_metrics(self):

        self.compute_adjusted_time = self.compute_adjusted_time()

        self.training_ball_distances, self.test_ball_distances = (
            self.get_F1_ball_distances()
        )

        self.F1_checkpoints = self.find_checkpoint_times()

        self.direction_match = self.get_direction_match()

        self.metrics = {
            "adjusted_time": self.compute_adjusted_time,
            "training_ball_distances": self.training_ball_distances,
            "test_ball_distances": self.test_ball_distances,
            "F1_checkpoints": self.F1_checkpoints,
            "direction_match": self.direction_match,
        }

    def compute_adjusted_time(self):
        """
        Compute adjusted time based on the fly's exit time if any, otherwise return NaN.
        """
        if self.tracking_data.exit_time is not None:
            flydata = self.tracking_data.flytrack.objects[0].dataset
            flydata["adjusted_time"] = flydata["time"] - self.tracking_data.exit_time
            return flydata["adjusted_time"]
        else:
            return None

    def get_F1_ball_distances(self):
        """
        Compute the Euclidean distances for the training and test ball data.

        Returns:
            tuple: The training and test ball data with Euclidean distances.
        """
        training_ball_data = None
        test_ball_data = None

        for ball_idx in range(0, len(self.tracking_data.balltrack.objects)):
            ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

            # Check if the ball is > or < 100 px away from the fly's initial x position
            if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                training_ball_data = ball_data
                training_ball_data["euclidean_distance"] = np.sqrt(
                    (
                        training_ball_data["x_centre"]
                        - training_ball_data["x_centre"].iloc[0]
                    )
                    ** 2
                    + (
                        training_ball_data["y_centre"]
                        - training_ball_data["y_centre"].iloc[0]
                    )
                    ** 2
                )
            else:
                test_ball_data = ball_data
                test_ball_data["euclidean_distance"] = np.sqrt(
                    (test_ball_data["x_centre"] - test_ball_data["x_centre"].iloc[0])
                    ** 2
                    + (test_ball_data["y_centre"] - test_ball_data["y_centre"].iloc[0])
                    ** 2
                )

        return training_ball_data, test_ball_data

    def find_checkpoint_times(self, distances=[10, 25, 35, 50, 60, 75, 90, 100]):
        """
        Find the times at which the ball reaches certain distances from its initial position.
        """
        _, test_ball_data = self.get_F1_ball_distances()

        checkpoint_times = {}

        for distance in distances:
            try:
                # Find the time at which the test ball reaches the distance
                checkpoint_time = test_ball_data.loc[
                    test_ball_data["euclidean_distance"] >= distance, "time"
                ].iloc[0]
                # Adjust the time by subtracting self.tracking_data.exit_time
                adjusted_time = checkpoint_time - self.tracking_data.exit_time
            except IndexError:
                # If the distance is not reached, set the checkpoint time to None
                adjusted_time = None

            checkpoint_times[f"{distance}"] = adjusted_time

        return checkpoint_times

    def get_direction_match(self):
        """
        For each of the flyball pair, check the success_direction metric and compare it with the success_direction of the other flyball pair.
        """

        success_directions = {}

        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                key = f"fly_{fly_idx}_ball_{ball_idx}"
                success_direction = self.fly.events_metrics[key]["success_direction"]
                success_directions[key] = success_direction

        if len(self.tracking_data.balltrack.objects) > 1:
            direction_1 = success_directions["fly_0_ball_0"]
            direction_2 = success_directions["fly_0_ball_1"]

            if direction_1 == direction_2:
                return "match"
            elif direction_1 == "both" or direction_2 == "both":
                return "partial_match"
            else:
                return "different"
        else:
            return None