#import Liabrary
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate data
np.random.seed(41)  # For reproducibility
cluster_1 = np.random.normal(loc=[0, 0], scale=0.3, size=(50, 2))
cluster_2 = np.random.normal(loc=[1, 1], scale=0.3, size=(50, 2))
cluster_3 = np.random.normal(loc=[1, 0], scale=0.3, size=(50, 2))
cluster_4 = np.random.normal(loc=[-1, -1], scale=0.3, size=(50, 2))

data = np.vstack((cluster_1, cluster_2, cluster_3, cluster_4))

# Helper function to plot data
def plot_data(points, colors, title="Data Plot"):
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Step 2: Plot the data without clustering
plot_data(data, 'grey', "Data without Clustering")


# K-Means Algorithm Implementation
def k_means(data, k=3, n_iters=100, nstart=25,random_state=41):
    best_clusters = None
    best_centroids = None    
    best_inertia = np.inf
    
    n_samples = data.shape[0]

    for _ in range(nstart):   # _ = not used variable(Temporary variable)
        # ================= EX1 ===========================
        # Randomly choose k data points as initial centroids
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = data[indices].copy()

        # Assign clusters based on closest centroid
        for _ in range(n_iters):
            # ================= EX2 ===========================
            distances = np.zeros((n_samples, k))
            for i in range(k):
                # Calculate the Euclidean distance from each data point to each centroid
                diff = data - centroids[i]
                distances[:, i] = np.sqrt(np.sum(diff**2, axis=1))  # Euclidean distance

            clusters = np.argmin(distances, axis=1)  # return index that min
            
            # ================= EX3 ===========================
            # Calculate new centroids from the means of the points
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = data[clusters == i]
                if cluster_points.size > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]  # Avoid empty cluster

            # Check for convergence
            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        # Calculate inertia (sum of squared distances to closest centroid)
        inertia = 0.0
        for i in range(k):
            cluster_data = data[clusters == i]
            diff = cluster_data - centroids[i]
            squared_diff = np.power(diff, 2)
            summed_squared_diff = np.sum(squared_diff)
            inertia += summed_squared_diff

        # Check if this result is the best one
        if inertia < best_inertia:
            best_inertia = inertia
            best_clusters = clusters.copy()
            best_centroids = centroids.copy()

    return best_clusters, best_centroids, best_inertia


# Step 3: Elbow Method to choose k
wcss = []
K_values = range(1, 11)  # Testing 1 to 10 clusters

for k in K_values:
    clusters, centroids, inertia = k_means(data, k=k, n_iters=100, nstart=10)
    wcss.append(inertia)  # inertia is our WCSS

# Plotting the WCSS to observe the 'Elbow'
plt.figure(figsize=(10, 8))
plt.plot(K_values, wcss, marker='o')
plt.title('Elbow Method For Optimal k (manual K-means)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()


# Step 4: Cluster the data with chosen k
clusters, centroids, inertia = k_means(data, k=4, nstart=90)

# Step 5: Plot the data with clusters
plot_data(data, clusters, "Data with Cluster Coloring")

# Plot centroids
plt.scatter(data[:, 0], data[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')  # Mark centroids
plt.title("Data with Centroids")
plt.show()
