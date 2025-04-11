import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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
            print(f"Feature columns: {features.columns.tolist()}")

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
        return self.umap_model.fit_transform(features)

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
        result_df = pd.concat(
            [
                pd.DataFrame(embeddings, columns=[f"UMAP{i+1}" for i in range(self.n_components)]),
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
