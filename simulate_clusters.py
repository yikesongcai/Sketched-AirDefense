import numpy as np

def simulate_clustering_new_logic(seed=2, num_users=41, num_byz=8, num_clusters=8, num_classes=10):
    # Match main.py initialization
    np.random.seed(seed)
    
    # In generate.py now logic:
    distance = np.random.rand(num_users)
    distance.sort() # New logic
    
    K = num_users - 1
    indices = list(range(K))
    
    # User i distance is distance[i]
    user_distances = distance[:K]
    
    # In defense.py: get_cluster_assignments sorts indices by user_distances[i]
    sorted_indices = sorted(indices, key=lambda i: user_distances[i])
    
    num_per_cluster = K // num_clusters # 40 // 8 = 5
    
    byz_indices = set(range(num_byz))
    
    clusters = []
    for c in range(num_clusters):
        start_idx = c * num_per_cluster
        end_idx = start_idx + num_per_cluster if c < num_clusters - 1 else K
        cluster_indices = sorted_indices[start_idx:end_idx]
        
        byz_count = sum(1 for idx in cluster_indices if idx in byz_indices)
        honest_count = len(cluster_indices) - byz_count
        clusters.append({
            'cluster_id': c,
            'total': len(cluster_indices),
            'byz': byz_count,
            'honest': honest_count,
            'indices': [int(idx) for idx in cluster_indices]
        })
        
    return clusters

if __name__ == "__main__":
    results = simulate_clustering_new_logic(num_byz=8)
    print(f"Byz=8, Clusters=8 Logic Result:")
    print(f"{'Cluster':<10} | {'Total':<10} | {'Byzantine':<10} | {'Honest':<10} | {'Byz Ratio':<10}")
    print("-" * 70)
    for c in results:
        ratio = c['byz'] / c['total']
        print(f"{c['cluster_id']:<10} | {c['total']:<10} | {c['byz']:<10} | {c['honest']:<10} | {ratio:<10.2f}")
