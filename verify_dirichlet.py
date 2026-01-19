import numpy as np
import argparse

def test_dirichlet_distribution(alpha=0.1, num_users=41, num_classes=10):
    print(f"Testing Dirichlet Non-IID with alpha={alpha}")
    
    # Simulate main.py logic
    non_iid_matrix = np.random.dirichlet([alpha] * num_classes, size=num_users-1)
    non_iid_p = non_iid_matrix.tolist()
    
    # Last user (server)
    temp = np.ones(num_classes) / num_classes
    non_iid_p.append(temp.tolist())
    
    # Show distribution for first few clients
    for i in range(min(5, num_users)):
        print(f"Client {i} distribution: {[f'{p:.2f}' for p in non_iid_p[i]]}")
        
    # Show server distribution
    print(f"Server (Client {num_users-1}) distribution: {[f'{p:.2f}' for p in non_iid_p[-1]]}")

if __name__ == "__main__":
    test_dirichlet_distribution(alpha=0.1) # Extreme non-iid
    print("\n" + "="*50 + "\n")
    test_dirichlet_distribution(alpha=100.0) # Almost iid
