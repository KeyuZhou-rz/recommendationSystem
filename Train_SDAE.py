import torch
import torch.optim as optim
import torch.nn as nn
from SDAE_model import SDAE_GCL
import numpy as np
import os
from PIL import Image
from RGBConvHSV import modify_hsv_img
from PatternFeatureExtractor import PatternFeatureExtractor, extract_pattern_features
CATEGORY_MAP = {
    'suit': 0, 'sweater': 1, 'padding': 2, 'shirt': 3, 'tee': 4, 
    'windbreak': 5, 'mountainwear': 6, 'fur': 7, 'hoodies': 8, 
    'jacket': 9, 'vest': 10
}

SEARCH_TERM_MAP = {
    'casual':0,
    'formal':1,
    'sporty':2,
    'chic':3
}
def fetch_data(dataset_dir, num_samples=None):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f'the path {dataset_dir} is not right.')
    
    # store image dir
    pattern_list, color_list, labels_list = [], [], []

    print("scanning directory...")
    
    for root, dir, filenames in os.walk(dataset_dir):
        folder_name = os.path.basename(root).lower()
        label_idx = -1

        for term, idx in SEARCH_TERM_MAP.items():
            if term in root:
                label_idx = idx
                break
        
        if label_idx == -1:
            continue
        
        print(f'Found style folder: {folder_name} -> Label:{label_idx}')
        for filename in filenames:
            if not filename.lower().endswith(('.png', '.jpg', 'jpeg')):
                continue

            full_path = os.path.join(root, filename)
            try:
                pattern_vec = extract_pattern_features(full_path)
                if pattern_vec is None:
                    continue

                # color vector
                color_vec = modify_hsv_img(full_path, display_result=False)

                pattern_list.append(pattern_vec)
                color_list.append(color_vec)
                labels_list.append(label_idx)

            except Exception as e:
                print(f'Failed to process {filename}:{e}')

    if len(pattern_list) == 0:
        raise ValueError("No valid images found.")
    

    pattern_matrix = np.array(pattern_list,dtype=np.float32)
    color_matrix = np.array(color_list, dtype=np.float32)

    p_mean = np.mean(pattern_matrix, axis=0)
    p_std = np.std(pattern_matrix,axis=0)
    # avoid division by zero
    p_std[p_std == 0] = 1.0
    pattern_norm = (pattern_matrix - p_mean) / p_std
    
    c_min = np.min(color_matrix,axis=0)
    c_max = np.max(color_matrix,axis=0)
    c_denom = c_max - c_min
    c_denom[c_denom == 0] = 1.0
    color_norm = (color_matrix - c_min) / c_denom

    final_matrix = np.concatenate([pattern_norm, color_norm], axis=1)

    x_train = torch.tensor(final_matrix, dtype=torch.float32)
    c_train = torch.tensor(np.array(labels_list), dtype=torch.long)


    return x_train, c_train


def train_model(dataset_dir):
    num_styles = len(SEARCH_TERM_MAP)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = SDAE_GCL(input_dim_pattern=2048, input_dim_color=20, category_dim=num_styles, latent_dim=64)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Loss Functions
    criterion_mse = nn.MSELoss()
    criterion_cat = nn.CrossEntropyLoss() # as loss func
    # hyper para
    lambda_gcl = 5.0

    x_train, c_train = fetch_data(dataset_dir)
    x_train = x_train.to(device)
    c_train = c_train.to(device)

    print(f"Data loaded. Shape: {x_train.shape}. Training on {num_styles} styles.")
    
    # training loop
    print("Training starts...")
    model.train()
    criterion_reconstruct = nn.MSELoss()
    criterion_category = nn.CrossEntropyLoss()

    for epoch in range(50):
        optimizer.zero_grad()
        # noise added as paper recommends
        noise = torch.randn_like(x_train) * 0.1
        x_noisy = x_train + noise

        x_hat, c_hat, latent = model(x_noisy)
        # calculate loss
        # loss 1: visual features

        loss_visual = criterion_reconstruct(x_hat,x_train)

        # loss 2: category correct?
        loss_category = criterion_category(c_hat,c_train)

        # Total loss
        total_loss = loss_visual + (lambda_gcl * loss_category)
        total_loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.4f} (Vis: {loss_visual:.4f}, Cat: {loss_category:.4f})")

    print("Traning completes.")
    return model
    
if __name__ == '__main__':
    dataset_path = 'data'
    if os.path.exists(dataset_path):
        trained_model = train_model(dataset_path)
        # save the outcome
        torch.save(trained_model.state_dict(),'SDAE_GCL_trained.pth')
        print("Model save to SDAE_GCL_trained.pth")

    else:
        print("Please set the correct dataset_path in script.")
        
