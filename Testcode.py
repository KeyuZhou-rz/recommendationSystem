from PatternFeatureExtractor import PatternFeatureExtractor, extract_pattern_features
from RGBConvHSV import modify_hsv_img
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


img_path = '/Users/rzzz/Desktop/Workspace/Projects/recommendation/recommendationSystem/Slice_24_04_16_USNews.jpg'

# HSV-based features
para_hsv = modify_hsv_img(img_path)

# Pattern features via class
'''
instance = PatternFeatureExtractor()
vec = instance.extract_pattern_features(img_path)
print(vec.shape)
'''

# Also test the standalone convenience function
vec2 = extract_pattern_features(img_path)
print(para_hsv,vec2)
