import numpy as np
import time

# Step 1: Traffic Flow Generation (including packet error rate)
def generate_traffic_flows(num_flows, traffic_model):
    flows = []
    for traffic_type, (arrival_rate, size_range, proportion) in traffic_model.items():
        num_type_flows = int(num_flows * proportion)
        for _ in range(num_type_flows):
            packet_size = np.random.randint(size_range[0], size_range[1])
            # Assign random QoS values based on traffic type
            latency = np.random.uniform(0.001, 1)  # Example: Random latency (in ms)
            bandwidth = packet_size / 1000  # Example: Bandwidth based on packet size
            packet_error_rate = np.random.uniform(0, 0.1)  # Random packet error rate (0-10%)
            flows.append([latency, bandwidth, packet_error_rate])
    return np.array(flows)

# Step 2: Normalize the QoS values (normalize to [0, 1] range, including packet error rate)
def normalize(qos_values):
    min_vals = np.min(qos_values, axis=0)
    max_vals = np.max(qos_values, axis=0)
    return (qos_values - min_vals) / (max_vals - min_vals), min_vals, max_vals

# Step 3: Initialize cluster centers (traffic types) with three QoS parameters
def initialize_cluster_centers():
    # Initial centers based on estimated average QoS requirements of different traffic types
    network_control_center = np.array([0.1, 0.9, 0.01])  # Example: low latency, high bandwidth, low error rate
    best_effort_center = np.array([0.9, 0.2, 0.05])  # Example: high latency, low bandwidth, higher error rate
    return np.array([network_control_center, best_effort_center])

# Step 4: Calculate Euclidean distance between a flow and the cluster center
def euclidean_distance(flow, center):
    return np.sqrt(np.sum((flow - center) ** 2))

# Step 5: Assign traffic flows to the closest cluster
def assign_to_cluster(flows, cluster_centers):
    assignments = []
    for flow in flows:
        distances = [euclidean_distance(flow, center) for center in cluster_centers]
        closest_cluster = np.argmin(distances)
        assignments.append(closest_cluster)
    return assignments

# Step 6: Update cluster centers based on assigned flows
def update_cluster_centers(flows, assignments, cluster_centers):
    new_centers = []
    for i in range(len(cluster_centers)):
        cluster_flows = [flows[j] for j in range(len(flows)) if assignments[j] == i]
        if cluster_flows:
            new_center = np.mean(cluster_flows, axis=0)
            new_centers.append(new_center)
        else:
            new_centers.append(cluster_centers[i])
    return np.array(new_centers)

# Step 7: Check convergence
def has_converged(old_centers, new_centers, threshold=0.001):
    distances = [euclidean_distance(old, new) for old, new in zip(old_centers, new_centers)]
    return all(distance < threshold for distance in distances)

# Improved K-means Clustering Process
def improved_k_means_clustering(flows, initial_cluster_centers, max_iterations=100):
    cluster_centers = initial_cluster_centers
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")
        assignments = assign_to_cluster(flows, cluster_centers)
        new_centers = update_cluster_centers(flows, assignments, cluster_centers)
        if has_converged(cluster_centers, new_centers):
            print("Converged!")
            break
        cluster_centers = new_centers
    return assignments, cluster_centers

# Step 8: Define 5QI Profiles (QoS Profiles) with packet error rate
def define_5qi_profiles():
    # Example 5QI profiles: [latency, bandwidth, packet_error_rate]
    profile_1 = np.array([0.05, 0.8, 0.02])  # Low latency, high bandwidth, low error rate (normalized)
    profile_2 = np.array([0.8, 0.2, 0.05])  # High latency, low bandwidth, higher error rate (normalized)
    profile_3 = np.array([0.5, 0.5, 0.04])  # Balanced profile with moderate error rate
    return np.array([profile_1, profile_2, profile_3])

# Step 9: Calculate distances between cluster centers and 5QI profiles
def calculate_5qi_distances(cluster_centers, qos_profiles):
    distances = []
    for center in cluster_centers:
        profile_distances = [euclidean_distance(center, profile) for profile in qos_profiles]
        distances.append(profile_distances)
    return np.array(distances)

# Step 10: Rough Set Theory - Assign profiles to approximation sets based on thresholds
def assign_profiles_to_sets(distances, min_threshold, relative_threshold):
    lower_approx_set = []
    upper_approx_set = []
    for distance_set in distances:
        lower = []
        upper = []
        for distance in distance_set:
            if distance <= min_threshold:
                lower.append(distance)
            elif distance <= relative_threshold:
                upper.append(distance)
        lower_approx_set.append(lower)
        upper_approx_set.append(upper)
    return lower_approx_set, upper_approx_set

# Step 11: Calculate membership degree for each profile
def calculate_membership_degrees(distances):
    membership_degrees = 1 / (distances ** 2)
    return membership_degrees

# Step 12: Assign best 5QI profile based on membership degree
def assign_best_5qi(lower_approx_set, upper_approx_set, membership_degrees):
    assignments = []
    for i, lower_set in enumerate(lower_approx_set):
        if lower_set:
            best_profile = np.argmax(membership_degrees[i])
            assignments.append(best_profile)
        else:
            if upper_approx_set[i]:
                best_profile = np.argmax(membership_degrees[i])
                assignments.append(best_profile)
            else:
                assignments.append(-1)  # No suitable profile
    return assignments

# Step 13: Simulate QoS Mapping Evaluation with time tracking
def evaluate_performance(traffic_flows, assignments, five_qi_assignments):
    # QoS mapping delay, throughput, and successful mapping rate calculation
    normalized_throughput = len(traffic_flows) / (640 * 10)  # Example: normalized throughput calculation
    successful_mapping_rate = np.mean([1 if profile >= 0 else 0 for profile in five_qi_assignments])
    print(f"Normalized Throughput: {normalized_throughput}")
    print(f"Successful Mapping Rate: {successful_mapping_rate * 100}%")

# Example traffic model based on Table 4
traffic_model = {
    "NetworkControl": (50, [50, 500], 0.02),
    "Isochronous": (500, [30, 100], 0.06),
    "Cycle": (50, [50, 1000], 0.06),
    "EventControl": (50, [100, 200], 0.05),
    "EventAlarm": (50, [100, 1500], 0.02),
    "Configuration": (50, [500, 1500], 0.08),
    "Video": (50, [1000, 1500], 0.15),
    "Audio": (50, [1000, 1500], 0.15),
    "BestEffort": (50, [30, 1500], 0.41),
}

# Simulate and evaluate for different numbers of traffic flows
for num_flows in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    print(f"\nEvaluating performance for {num_flows} traffic flows:")
    # Generate traffic flows
    traffic_flows = generate_traffic_flows(num_flows, traffic_model)
    # Normalize the flows
    normalized_flows, min_vals, max_vals = normalize(traffic_flows)
    # Initialize cluster centers and run IKC-RQM
    initial_centers = initialize_cluster_centers()
    start_mapping_time = time.time()
    assignments, final_centers = improved_k_means_clustering(normalized_flows, initial_centers)

    # Start mapping time tracking after cluster assignment

    # Define 5QI profiles and assign using Rough Set Theory
    qos_profiles = define_5qi_profiles()
    distances = calculate_5qi_distances(final_centers, qos_profiles)
    lower_approx_set, upper_approx_set = assign_profiles_to_sets(distances, min_threshold=0.05, relative_threshold=1.5)
    membership_degrees = calculate_membership_degrees(distances)
    five_qi_assignments = assign_best_5qi(lower_approx_set, upper_approx_set, membership_degrees)
    end_mapping_time = time.time()

    # Evaluate performance
    evaluate_performance(traffic_flows, assignments, five_qi_assignments)
    print(f"Total QoS Mapping Delay: {end_mapping_time - start_mapping_time} seconds\n")