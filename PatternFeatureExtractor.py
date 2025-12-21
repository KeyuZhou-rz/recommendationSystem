import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class PatternFeatureExtractor:
    def __init__(self):
        '''
        1. Load a pre-trained ResNet50 model
        2. Download weights from image net
        '''
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)

        # remove the last layer classifying dog vs cat

        self.layer = self.model._models.get('avgpool')
        self.model.eval() # evaluation mode

        # Define standard transform expected by Resnet

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])


        def get_vector(self,img_path):
            try:
                input_img = Image.open(img_path).convert('RGB')
            except:
                raise FileNotFoundError(f'the path {img_path} is wrong.')
            # preprocess image
            input_tensor = self.preporcess(input_img)
            input_batch = input_tensor.unsqueeze(0)

            # Run the model
            with torch.no_grad(): # Don't calculate gradients (saves memory)
            # We hook the output of the layer we want
            # (In ResNet, simply running the full model without the fc layer is complex, 
            # so usually we just extract from the backbone if using a library, 
            # but here is the manual "Forward Hook" way or the "modules" way.
            # SIMPLEST WAY: Use the model as a feature extractor directly):

                features = self.model(input_batch) 
            # WAIT: The default model returns class probabilities.
            # We must replace the final 'fc' layer with an Identity to get features.
            
            return features.flatten().numpy()

# --- CORRECTED INITIALIZATION FOR FEATURE EXTRACTION ---
# The cleanest way to get the feature vector in PyTorch:
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Replace the final classification layer (Linear) with an empty layer
resnet.fc = nn.Identity()
resnet.eval()

def extract_pattern_features(image_path, model=resnet):
    """
    Returns a numpy array of shape (2048,) representing the pattern/structure.
    """
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
            
        return features.flatten().numpy()
        
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

# Test
if __name__ == "__main__":
    '''
    path = "/home/ke_yu-zhou005/recommendationSystem/Slice_24_04_16_USNews.jpg"
    vec = extract_pattern_features(path)
    print(f"Pattern Vector Shape: {vec.shape}") # Should be (2048,)
    '''
    pass