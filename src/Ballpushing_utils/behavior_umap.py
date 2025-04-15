import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA, PCA
import umap
import os
from Ballpushing_utils import dataset


class BehaviorUMAP:
    def __init__(
        self,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        explained_variance_threshold=0.95,
        n_clusters=10,
        filter_features=True,
    ):
        """
        Initialize the BehaviorUMAP class.

        Args:
            n_neighbors (int): Number of neighbors for UMAP.
            min_dist (float): Minimum distance between points in UMAP.
            n_components (int): Number of dimensions for UMAP output.
            explained_variance_threshold (float): Threshold for explained variance in PCA.
            n_clusters (int): Number of clusters for KMeans.
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.explained_variance_threshold = explained_variance_threshold
        self.n_clusters = n_clusters
        self.umap_model = None
        self.kmeans = None
        self.filter_features = filter_features

    def prepare_features(self, data, feature_groups=None, include_ball=False):
        """
        Prepare features for UMAP based on selected feature groups.

        Args:
            data (pd.DataFrame): Input dataset.
            feature_groups (list): List of feature groups to include.
            include_ball (bool): Whether to include ball-related features.

        Returns:
            np.ndarray: Processed feature matrix.
            pd.DataFrame: Metadata for the dataset.
        """
        # Feature selection configuration
        feature_config = {
            "tracking": [r"_frame\d+_x$", r"_frame\d+_y$"],
            "frame": [r"_frame\d+_velocity$", r"_frame\d+_angle$"],
            "statistical": [r"_mean$", r"_std$", r"_skew$", r"_kurt$"],
            "fourier": [r"_dom_freq$", r"_dom_freq_magnitude$"],
        }

        # Create combined regex pattern
        regex_parts = []

        if feature_groups is None:
            feature_groups = list(feature_config.keys())

        for group in feature_groups:
            regex_parts.extend(feature_config.get(group, []))

        feature_pattern = "|".join(regex_parts)

        # Extract features and metadata
        feature_columns = data.filter(regex=feature_pattern).columns.tolist()

        if not include_ball:
            feature_columns = [col for col in feature_columns if "centre" not in col]

        if len(feature_columns) == 0:
            raise ValueError("No features found matching the selected feature groups")

        features = data[feature_columns]
        metadata = data.drop(columns=feature_columns, errors="ignore")

        if self.filter_features:
            # Filter features based on variance
            feature_variance = features.var(axis=0)
            selected_features = feature_variance[feature_variance > 0.01].index.tolist()
            features = features[selected_features]
            # Filter features based on correlation
            corr_matrix = features.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            features = features.drop(columns=to_drop, errors="ignore")
            print(f"Filtered features: {len(to_drop)} features removed based on correlation.")
            print(f"Remaining features after filtering: {len(features.columns)}")
        else:
            print("Feature filtering is disabled. All features will be used.")
            print(f"Total features selected: {len(features.columns)}")
            #print(f"Feature columns: {features.columns.tolist()}")

        return features, metadata

    def apply_pca(self, features):
        """
        Apply PCA to reduce dimensionality of features.

        Args:
            features (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: PCA-transformed features.
        """
        print("Applying PCA...")
        pca = PCA(n_components=min(features.shape), svd_solver="randomized").fit(features)
        n_components_pca = np.argmax(pca.explained_variance_ratio_.cumsum() >= self.explained_variance_threshold) + 1
        print(f"Reducing to {n_components_pca} components based on explained variance threshold.")
        return pca.transform(features)[:, :n_components_pca]

    def fit_umap(self, features):
        """
        Fit UMAP to the features.

        Args:
            features (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: UMAP embeddings.
        """
        print("Running UMAP...")
        self.umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric="euclidean",
            random_state=42,
        )
        return self.umap_model.fit_transform(features, ensure_all_finite="allow-nan")

    def cluster_embeddings(self, embeddings):
        """
        Cluster UMAP embeddings using KMeans.

        Args:
            embeddings (np.ndarray): UMAP embeddings.

        Returns:
            np.ndarray: Cluster labels.
        """
        print("Running KMeans clustering...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        return self.kmeans.fit_predict(embeddings)

    def generate_umap_and_clusters(self, data, feature_groups=None, include_ball=False, use_pca=False, savepath=None):
        """
        Generate UMAP embeddings and cluster labels for the dataset.

        Args:
            data (pd.DataFrame): Input dataset.
            feature_groups (list): List of feature groups to include.
            include_ball (bool): Whether to include ball-related features.
            use_pca (bool): Whether to apply PCA before UMAP.
            savepath (str): Path to save the resulting dataset with UMAP and cluster labels.

        Returns:
            pd.DataFrame: Dataset with UMAP embeddings and cluster labels.
        """
        # Prepare features and metadata
        features, metadata = self.prepare_features(data, feature_groups, include_ball)

        # Apply PCA if enabled
        if use_pca:
            features = self.apply_pca(features)

        # Fit UMAP
        embeddings = self.fit_umap(features)

        # Cluster embeddings
        cluster_labels = self.cluster_embeddings(embeddings)

        # Compile results
        umap_columns = [f"UMAP{i+1}" for i in range(self.n_components)]  # Dynamically generate column names
        result_df = pd.concat(
            [
                pd.DataFrame(embeddings, columns=umap_columns),  # Use dynamic column names
                metadata.reset_index(drop=True),
                pd.DataFrame({"cluster": cluster_labels}),
            ],
            axis=1,
        )

        # Save the dataset if savepath is provided
        if savepath:
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            result_df.to_feather(savepath)
            print(f"Dataset with UMAP and cluster labels saved to {savepath}")

        return result_df

    def generate_cluster_videos(self, Umap, interaction_data, output_dir, best_disp=False):
        """
        Generate grid videos for each cluster.

        Args:
            Umap (pd.DataFrame): DataFrame containing UMAP embeddings and cluster labels.
            interaction_data (pd.DataFrame): DataFrame containing interaction data.
            output_dir (str): Directory to save the generated videos.
            best_disp (bool): Whether to select events with the highest displacement.
        """
        # Configuration parameters
        MAX_CELL_WIDTH = 96   # Maximum width for grid cells
        MAX_CELL_HEIGHT = 516  # Maximum height for grid cells
        MAX_OUTPUT_WIDTH = 3840
        MAX_OUTPUT_HEIGHT = 2160
        FPS = 5
        CODEC = "mp4v"

        def resize_with_padding(frame, target_w, target_h):
            """Resize frame while maintaining aspect ratio with padding."""
            h, w = frame.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(frame, (new_w, new_h))
            pad_w = target_w - new_w
            pad_h = target_h - new_h

            # Add equal padding on both sides
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Add unique identifiers to both datasets
        Umap["unique_id"] = Umap["fly"].astype(str) + "_" + Umap["event_type"] + "_" + Umap["event_id"].astype(str)
        interaction_data["unique_id"] = interaction_data["fly"].astype(str) + "_" + interaction_data["event_type"] + "_" + interaction_data["event_id"].astype(str)

        # Loop over all clusters
        for cluster_id in Umap["cluster"].unique():
            cluster_data = Umap[Umap["cluster"] == cluster_id]
            cluster_interactions = interaction_data[interaction_data["unique_id"].isin(cluster_data["unique_id"])]

            # Calculate frame ranges for each unique_id
            frame_ranges = (cluster_interactions
                            .groupby('unique_id')['frame']
                            .agg(frame_start=('min'), frame_end=('max'))
                            .reset_index())

            # Merge with path information
            event_metadata = (cluster_interactions[['unique_id', 'flypath']]
                              .drop_duplicates()
                              .merge(frame_ranges, on='unique_id'))

            # Add raw_displacement from Umap to event_metadata
            event_metadata = event_metadata.merge(
                Umap[['unique_id', 'raw_displacement']],
                on='unique_id',
                how='left'
            )

            # Calculate grid layout based on max output dimensions
            cols = MAX_OUTPUT_WIDTH // MAX_CELL_HEIGHT  # Note the swapped dimensions
            rows = MAX_OUTPUT_HEIGHT // MAX_CELL_WIDTH  # Note the swapped dimensions
            max_events = cols * rows

            # Select events based on the best_disp argument
            if len(event_metadata) > max_events:
                if best_disp:
                    # Sort by raw_displacement in descending order and pick the top events
                    event_metadata = event_metadata.sort_values(by='raw_displacement', ascending=False).head(max_events)
                else:
                    # Randomly sample events
                    event_metadata = event_metadata.sample(max_events, random_state=42)

            # Initialize frame storage and video metadata
            frames_dict = {}
            max_duration = 0
            valid_events = 0

            # Process videos in optimized groups
            for flypath, group in event_metadata.groupby('flypath'):
                video_files = list(Path(flypath).glob("*.mp4"))
                video_file = next((vf for vf in video_files if "_preprocessed" not in vf.stem), None)

                if not video_file:
                    print(f"Skipping {flypath} - no suitable MP4 found")
                    continue

                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    print(f"Couldn't open {video_file}")
                    continue

                # Process all events from this video
                for _, event in group.iterrows():
                    try:
                        start = int(event['frame_start'])
                        end = int(event['frame_end'])
                        if start > end:
                            print(f"Invalid frames for {event['unique_id']}")
                            continue

                        # Read event frames with boundary checks
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                        frames = []
                        for _ in range(end - start + 1):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            # Rotate frame 90° clockwise
                            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                            # Convert color space and resize with padding
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = resize_with_padding(frame, MAX_CELL_HEIGHT, MAX_CELL_WIDTH)  # Note the swapped dimensions
                            frames.append(frame)

                        if frames:
                            frames_dict[event['unique_id']] = frames
                            max_duration = max(max_duration, len(frames))
                            valid_events += 1

                    except Exception as e:
                        print(f"Error processing {event['unique_id']}: {str(e)}")

                cap.release()

            # Early exit if no valid events
            if valid_events == 0:
                print(f"No processable events found for cluster {cluster_id}")
                continue

            # Pad all clips to max duration with black frames
            for uid in frames_dict:
                frames = frames_dict[uid]
                if len(frames) < max_duration:
                    padding = [np.zeros((MAX_CELL_WIDTH, MAX_CELL_HEIGHT, 3), dtype=np.uint8)] * (max_duration - len(frames))  # Note the swapped dimensions
                    frames_dict[uid] = frames + padding

            # Final output dimensions
            output_size = (cols * MAX_CELL_HEIGHT, rows * MAX_CELL_WIDTH)  # Note the swapped dimensions

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*CODEC)
            if best_disp:
                output_path = Path(output_dir) / f"cluster_{cluster_id}_video_best_disp.mp4"
            else:
                output_path = Path(output_dir) / f"cluster_{cluster_id}_video.mp4"
            out = cv2.VideoWriter(str(output_path), fourcc, FPS, output_size)

            # Generate grid frames
            for frame_idx in range(max_duration):
                grid = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

                for idx, (uid, frames) in enumerate(frames_dict.items()):
                    if frame_idx >= len(frames):
                        continue

                    row = idx // cols
                    col = idx % cols

                    # Calculate position
                    x = col * MAX_CELL_HEIGHT  # Note the swapped dimensions
                    y = row * MAX_CELL_WIDTH  # Note the swapped dimensions

                    # Place frame in grid cell
                    grid[y:y+MAX_CELL_WIDTH, x:x+MAX_CELL_HEIGHT] = frames[frame_idx]  # Note the swapped dimensions

                out.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

            out.release()
            print(f"Successfully created grid video for cluster {cluster_id} at {output_path}")
