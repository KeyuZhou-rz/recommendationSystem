import torch
import matplotlib.pyplot as plt
import numpy as np
from SDAE_model import SDAE_GCL
from Train_SDAE import fetch_data, SEARCH_TERM_MAP

def visualize(data_path):
    # 1. Load Data and Model
    x_data, c_data = fetch_data(data_path)
    model = SDAE_GCL(category_dim=4)
    model.load_state_dict(torch.load("SDAE_GCL_trained.pth"))
    model.eval()
    
    # 2. Get Coordinates (Inference)
    with torch.no_grad():
        _, _, latent_vectors = model(x_data)
        coordinates = latent_vectors.numpy()
        labels = c_data.numpy()

    # 3. Plot
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple']
    style_names = list(SEARCH_TERM_MAP.keys())
    
    for i in range(4):
        # Select points belonging to style i
        mask = (labels == i)
        plt.scatter(coordinates[mask, 0], coordinates[mask, 1], 
                    c=colors[i], label=style_names[i], alpha=0.7)
    
    plt.title("Fashion Image Scale Space (Learned from Search Terms)")
    plt.xlabel("Dimension 1 (e.g., Warm/Cool)")
    plt.ylabel("Dimension 2 (e.g., Hard/Soft)")
    plt.legend()
    plt.grid(True)
    plt.savefig("distribution_map.png")
    plt.show()

if __name__ == "__main__":
    visualize("data")