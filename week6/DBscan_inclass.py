import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=41)
X = StandardScaler().fit_transform(X)

# Plotting before clustering
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], color='gray', marker='o')
plt.title('Sample Data Before Clustering')
plt.grid(True)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()




def scratch_dbscan(data, eps, min_samples):
    n_points = len(data)
    labels = np.full(n_points, -1)  # Initialize all labels as -1 (noise)
    core_points = np.zeros(n_points, dtype=bool)
    
    # Step 1: Find all core points
    for i in range(n_points):
        # Compute distances from point i to all other points
        distances = np.linalg.norm(data - data[i], axis=1)
        # Count the number of neighbors within eps (including the point itself)
        neighbors = np.where(distances <= eps)[0]
        if len(neighbors) >= min_samples:
            core_points[i] = True

    # Step 2: Cluster core points
    cluster_id = 0
    visited = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        if core_points[i] and not visited[i]:
            # Start a new cluster
            labels[i] = cluster_id
            visited[i] = True
            # Initialize the queue with the current core point
            queue = [i]
            while queue:
                current_point = queue.pop(0)
                # Compute distances from current_point to all other points
                distances = np.linalg.norm(data - data[current_point], axis=1)
                neighbors = np.where(distances <= eps)[0]
                for neighbor in neighbors:
                    if core_points[neighbor] and not visited[neighbor]:
                        labels[neighbor] = cluster_id
                        visited[neighbor] = True
                        queue.append(neighbor)
            cluster_id += 1

    # Step 3: Assign non-core points
    for i in range(n_points):
        if not core_points[i]:
            # Compute distances from point i to all other points
            distances = np.linalg.norm(data - data[i], axis=1)
            neighbors = np.where(distances <= eps)[0]
            # Get clusters of neighboring core points
            neighbor_clusters = set()
            # Loop through each neighbor in the list of neighbors
            for neighbor in neighbors:
                # Check if the neighbor is a core point
                if core_points[neighbor]:
                    # Get the cluster label of this core point
                    label = labels[neighbor]
                    # Add the label to the set of neighbor clusters
                    neighbor_clusters.add(label)
                    
            if neighbor_clusters:
                # Randomly select one cluster
                labels[i] = np.random.choice(list(neighbor_clusters))
            else:
                # Remain as noise (-1)
                pass

    return labels


# Run the Modified DBSCAN
my_labels = scratch_dbscan(X, eps=0.2, min_samples=10)

# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=my_labels, cmap='viridis', marker='o')
plt.title('Modified DBSCAN Implementation from Scratch')
plt.grid(True)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

