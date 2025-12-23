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

def fetch_data(dataset_dir,num_samples=None):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f'the path {dataset_dir} is not right.')
    
    # store image dir
    image = []
    features_list, labels_list = [],[]

    for dirpath, _, filenames in os.walk(dataset_dir):

        for filename in filenames:
            if not filename.lower().endwith(('.png','.jpg','jpeg')):
                continue

            full_path = os.path.join(dirpath,filename)
            folder_name = os.path.basename(dirpath).lower()
            if folder_name not in CATEGORY_MAP:
                continue
            label_idx = CATEGORY_MAP[folder_name]

            try:
                pattern_vec = extract_pattern_features(full_path)
                if pattern_vec is None:
                    continue

                # color vector
                color_vec = modify_hsv_img(full_path)
                combined_vec = np.concatenate([pattern_vec, color_vec])

                features_list.append(combined_vec)
                labels_list.append(label_idx)

            except Exception as e:
                print(f'Failed to process {filename}:{e}')

    
    x_train = torch.tensor(np.array(features_list),dtype=torch.float32)
    c_train = torch.tensor(np.array(labels_list),dtype=torch.long)

    return x_train, c_train
    



def train_model():
    model = SDAE_GCL(input_dim=2068, category_dim=11)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Loss Function
    criterion_mse = nn.MSELoss()
    criterion_cat = nn.CrossEntropyLoss() # as loss func
    # hyper para
    lambda_gcl = 0.5

    x_train, c_train = fetch_data()

    # training loop
    print("Training starts...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        # noise added as paper recommends
        noise = torch.randn_like(x_train) * 0.1
        x_noisy = x_train + noise

        x_hat, c_hat, latent = model(x_noisy)
        # calculate loss
        # loss 1: visual features

        loss_visual = criterion_mse(x_hat,x_train)

        # loss 2: category correct?
        c_target_indices = torch.argmax(c_train, dim=1)
        loss_category = criterion_cat(c_hat,c_target_indices)

        # Total loss
        total_loss = loss_visual + (lambda_gcl * loss_category)
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.4f} (Vis: {loss_visual:.4f}, Cat: {loss_category:.4f})")

        print("Traning completes.")
        return model
    
    if __name__ == '__main__':
        train_model()
