import numpy as np

def simulate_clustering_exact(seed=2, num_users=41, num_byz=12, num_clusters=8, num_classes=10):
    # Match main.py initialization
    np.random.seed(seed)
    
    # Simulate random calls in main.py before generate_clients
    # Line 61-68 in main.py loop over num_users (41) - wait, log says 40 clients + 1 server
    # Loop for i in range(args.num_users-1): 40 iterations
    for i in range(num_users - 1):
        # Line 63: temp=np.random.randint(50,60, size=args.num_classes)
        # This consumes num_classes (10) random integers per client
        _ = np.random.randint(50, 60, size=num_classes)
        
    # Now call generate_clients logic
    # Line 7 in generate.py
    distance = np.random.rand(num_users)
    
    K = num_users - 1
    indices = list(range(K))
    
    # Byzantine indices are 0 to num_byz-1
    byz_indices = set(range(num_byz))
    
    # Sort indices by distance (indices 0 to 39)
    sorted_indices = sorted(indices, key=lambda i: distance[i])
    
    num_per_cluster = K // num_clusters # 40 // 8 = 5
    
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
    results = simulate_clustering_exact()
    print(f"{'Cluster':<10} | {'Total':<10} | {'Byzantine':<10} | {'Honest':<10} | {'Byz Ratio':<10}")
    print("-" * 70)
    for c in results:
        ratio = c['byz'] / c['total']
        print(f"{c['cluster_id']:<10} | {c['total']:<10} | {c['byz']:<10} | {c['honest']:<10} | {ratio:<10.2f}")
    
    total_byz = sum(c['byz'] for c in results)
    print(f"\nTotal Byzantine clients: {total_byz}")
