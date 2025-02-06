import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename):
    """Load 2D points from text file."""
    return np.loadtxt(filename)

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centers(data, k):
    """Initialize k cluster centers randomly within the data bounds."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    centers = np.random.uniform(min_vals, max_vals, (k, 2))
    return centers

def find_nearest_center(point, centers):
    """Find the index of the nearest center to a point."""
    distances = [euclidean_distance(point, center) for center in centers]
    return np.argmin(distances)

def find_confusing_points(data, centers, labels):
    """
    Find points that are nearly equidistant to multiple centers
    (difference in distances less than threshold)
    """
    threshold = 0.05 
    confusing_points = []
    
    for i, point in enumerate(data):
        distances = [euclidean_distance(point, center) for center in centers]
        sorted_distances = np.sort(distances)
        if (sorted_distances[1] - sorted_distances[0]) < threshold:
            confusing_points.append(i)
            
    return confusing_points

def kmeans_clustering(data, k=3, max_iters=100):
    """
    Perform k-means clustering on the data.
    Returns centers, labels, and confusing points.
    """
    # Initialize centers
    centers = initialize_centers(data, k)
    
    for iteration in range(max_iters):

        old_labels = np.array([find_nearest_center(point, centers) for point in data])
        
        new_centers = np.array([data[old_labels == i].mean(axis=0) if np.sum(old_labels == i) > 0 
                              else centers[i] for i in range(k)])
        
        if np.all(centers == new_centers):
            break
            
        centers = new_centers
    
    labels = np.array([find_nearest_center(point, centers) for point in data])
    
    # Find confusing points
    confusing_points = find_confusing_points(data, centers, labels)
    
    return centers, labels, confusing_points

def plot_clusters(data, centers, labels, confusing_points, run_number):
    """Plot the clusters, centers, and confusing points."""

    os.makedirs('./PLOTS', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    for i in range(len(centers)):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], 
                   label=f'Cluster {i+1}', alpha=0.6)
    
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', 
               s=200, label='Centers')
    
    if confusing_points:
        for point_idx in confusing_points:
            point = data[point_idx]
            box_size = 0.01 
            plt.gca().add_patch(plt.Rectangle(
                (point[0] - box_size/2, point[1] - box_size/2),
                box_size, box_size,
                fill=False, color='black', linewidth=1,
                label='Confusing Points' if point_idx == confusing_points[0] else ""
            ))
    
    plt.title(f'K-Means Clustering - Run {run_number+1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(f'./PLOTS/run_{run_number+1}_clusters.png')
    plt.close()

def main():
    # Load data
    data = load_data('/Users/prabhavsingh/Documents/CLASSES/Spring25/Speech/520.666-S25/HW1/hw1-data 2-1.txt')
    
    # Perform multiple runs
    n_runs = 6
    all_centers = []
    all_confusing_points = []
    
    for run in range(n_runs):
        print(f"\nRun {run+1}:")
        centers, labels, confusing_points = kmeans_clustering(data)
        all_centers.append(centers)
        all_confusing_points.append(confusing_points)
        
        # Print centers
        print("Cluster Centers:")
        for i, center in enumerate(centers):
            print(f"Cluster {i+1}: ({center[0]:.4f}, {center[1]:.4f})")
        
        if confusing_points:
            print(f"Number of confusing points: {len(confusing_points)}")
        
        # Plot results
        plot_clusters(data, centers, labels, confusing_points, run)
    
    print("\nAll runs completed. Plots saved in ./PLOTS directory")

if __name__ == "__main__":
    main()