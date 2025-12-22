import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class PatternFeatureExtractor:
    """Wrapper around a ResNet50 feature extractor.

    Usage:
      inst = PatternFeatureExtractor()
      vec = inst.extract_pattern_features(path)
    """

    def __init__(self, device=None):
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load pretrained resnet50 and replace fc with identity to get a 2048-dim vector.
        try:
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
            # for mac, a piece of ssl should be in Testcode
        except Exception as e:
            print(f"Warning: failed to load pretrained ResNet50 weights ({e}). Using uninitialized model.")
            self.model = models.resnet50(weights=None)
        # Just output a 2048 raw features
        self.model.fc = nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_pattern_features(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception:
            raise FileNotFoundError(f'the path {image_path} is wrong.')

        input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(input_tensor)

        return features.squeeze(0).cpu().numpy()


# Convenience standalone function for backward compatibility
def extract_pattern_features(image_path, model=None):
    """Return a 2048-dim numpy vector for the image_path using an internal ResNet.

    If model is provided it should be a torch.nn.Module that maps a batch tensor to
    a (batch, 2048) feature tensor.
    """
    if model is None:
        # Attempt to load pretrained weights, but fall back to no-weights if necessary
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception as e:
            print(f"Warning: failed to load pretrained ResNet50 weights for standalone function ({e}). Using uninitialized model.")
            model = models.resnet50(weights=None)
        model.fc = nn.Identity()
        model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = model(input_tensor)
        return features.squeeze(0).numpy()
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None