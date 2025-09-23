import numpy as np


class F1Metrics:
    def __init__(self, tracking_data):
        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        self.metrics = {}
        self.compute_metrics()

    def compute_metrics(self):

        self.adjusted_time_metrics = self.compute_adjusted_time()

        self.training_ball_distances, self.test_ball_distances = self.get_F1_ball_distances()

        self.F1_checkpoints = self.find_checkpoint_times()

        self.direction_match = self.get_direction_match()

        self.metrics = {
            "adjusted_time": self.adjusted_time_metrics,
            "training_ball_distances": self.training_ball_distances,
            "test_ball_distances": self.test_ball_distances,
            "F1_checkpoints": self.F1_checkpoints,
            "direction_match": self.direction_match,
        }

    def compute_adjusted_time(self):
        """
        Compute adjusted time metrics for F1 experiments.

        Returns:
            dict: Adjusted time metrics
        """
        training_ball_data, test_ball_data = self.get_F1_ball_distances()
        checkpoints = self.find_checkpoint_times()

        metrics = {}

        # Time to reach test ball (adjusted for training period)
        if "first_test_contact" in checkpoints:
            test_contact_time = checkpoints["first_test_contact"]
            metrics["time_to_test_ball"] = test_contact_time / self.fly.experiment.fps

        # Time to reach training ball (if it exists)
        if "first_training_contact" in checkpoints:
            training_contact_time = checkpoints["first_training_contact"]
            metrics["time_to_training_ball"] = training_contact_time / self.fly.experiment.fps

        # Efficiency metric (distance vs time to test ball)
        if "first_test_contact" in checkpoints and test_ball_data is not None:
            test_contact_frame = checkpoints["first_test_contact"]
            # Calculate total fly distance traveled
            fly_coords = self.fly.flyball_positions[["x_thorax", "y_thorax"]].values[:test_contact_frame]
            if len(fly_coords) > 1:
                fly_distances = np.sqrt(np.sum(np.diff(fly_coords, axis=0) ** 2, axis=1))
                total_fly_distance = np.sum(fly_distances)
                # Initial distance between fly and test ball
                fly_start = self.fly.flyball_positions[["x_thorax", "y_thorax"]].iloc[0].values
                ball_start = test_ball_data[["x_centre", "y_centre"]].iloc[0].values
                direct_distance = np.sqrt(np.sum((ball_start - fly_start) ** 2))

                efficiency = direct_distance / (total_fly_distance + 1e-8)  # Avoid division by zero
                metrics["path_efficiency_to_test"] = efficiency

        return metrics

    def get_F1_ball_distances(self):
        """
        Compute the Euclidean distances for the training and test ball data.

        Returns:
            tuple: The training and test ball data with Euclidean distances.
        """
        training_ball_data = None
        test_ball_data = None

        num_balls = len(self.tracking_data.balltrack.objects)

        if num_balls == 1:
            # Single ball experiment: the ball is always the test ball
            ball_data = self.tracking_data.balltrack.objects[0].dataset.copy()
            ball_data["euclidean_distance"] = np.sqrt(
                (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
                + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
            )
            test_ball_data = ball_data

        elif num_balls == 2:
            # Two ball experiment: assign based on distance from fly's start position
            for ball_idx in range(0, num_balls):
                ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset.copy()

                # Add euclidean distance column
                ball_data["euclidean_distance"] = np.sqrt(
                    (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
                    + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
                )

                # Check if the ball is > or < 100 px away from the fly's initial x position
                if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                    training_ball_data = ball_data
                else:
                    test_ball_data = ball_data

        return training_ball_data, test_ball_data

    def find_checkpoint_times(self):
        """
        Find checkpoint times in the F1 experiment.

        Returns:
            dict: Dictionary containing checkpoint times
        """
        training_ball_data, test_ball_data = self.get_F1_ball_distances()

        # Use test ball data to find checkpoints (test ball should always exist)
        checkpoints = {}

        if test_ball_data is not None:
            # Calculate distances between fly and test ball over time
            fly_coords = self.fly.flyball_positions[["x_thorax", "y_thorax"]].values
            ball_coords = test_ball_data[["x_centre", "y_centre"]].values

            # Make sure arrays are same length
            min_len = min(len(fly_coords), len(ball_coords))
            fly_coords = fly_coords[:min_len]
            ball_coords = ball_coords[:min_len]

            fly_ball_distances = np.sqrt(np.sum((fly_coords - ball_coords) ** 2, axis=1))
            contact_threshold = 50  # pixels

            first_contact_indices = np.where(fly_ball_distances < contact_threshold)[0]
            if len(first_contact_indices) > 0:
                checkpoints["first_test_contact"] = first_contact_indices[0]

        # If training ball exists, find training-related checkpoints
        if training_ball_data is not None:
            # Calculate distances between fly and training ball over time
            fly_coords = self.fly.flyball_positions[["x_thorax", "y_thorax"]].values
            ball_coords = training_ball_data[["x_centre", "y_centre"]].values

            # Make sure arrays are same length
            min_len = min(len(fly_coords), len(ball_coords))
            fly_coords = fly_coords[:min_len]
            ball_coords = ball_coords[:min_len]

            fly_ball_distances = np.sqrt(np.sum((fly_coords - ball_coords) ** 2, axis=1))
            contact_threshold = 50  # pixels

            first_contact_indices = np.where(fly_ball_distances < contact_threshold)[0]
            if len(first_contact_indices) > 0:
                checkpoints["first_training_contact"] = first_contact_indices[0]

        return checkpoints

    def get_direction_match(self):
        """
        Calculate direction matching between fly movement and ball positions.

        Returns:
            dict: Direction matching metrics
        """
        training_ball_data, test_ball_data = self.get_F1_ball_distances()

        fly_coords = self.fly.flyball_positions[["x_thorax", "y_thorax"]].values

        # Calculate fly movement vectors
        fly_movement = np.diff(fly_coords, axis=0)

        # Calculate direction towards each ball
        direction_metrics = {}

        # Direction towards training ball (only if it exists)
        if training_ball_data is not None:
            training_ball_coords = training_ball_data[["x_centre", "y_centre"]].values[
                :-1
            ]  # Remove last frame to match movement vectors
            fly_to_training = training_ball_coords - fly_coords[:-1]

            # Normalize vectors
            fly_movement_norm = fly_movement / (np.linalg.norm(fly_movement, axis=1, keepdims=True) + 1e-8)
            fly_to_training_norm = fly_to_training / (np.linalg.norm(fly_to_training, axis=1, keepdims=True) + 1e-8)

            # Calculate dot product (cosine similarity)
            direction_match_training = np.sum(fly_movement_norm * fly_to_training_norm, axis=1)
            direction_metrics["training_direction_match"] = np.mean(direction_match_training)

        # Direction towards test ball (should always exist)
        if test_ball_data is not None:
            test_ball_coords = test_ball_data[["x_centre", "y_centre"]].values[:-1]
            fly_to_test = test_ball_coords - fly_coords[:-1]

            # Normalize vectors
            fly_movement_norm = fly_movement / (np.linalg.norm(fly_movement, axis=1, keepdims=True) + 1e-8)
            fly_to_test_norm = fly_to_test / (np.linalg.norm(fly_to_test, axis=1, keepdims=True) + 1e-8)

            direction_match_test = np.sum(fly_movement_norm * fly_to_test_norm, axis=1)
            direction_metrics["test_direction_match"] = np.mean(direction_match_test)

        return direction_metrics
