import cv2
import numpy as np
from sklearn.cluster import KMeans


def modify_hsv_img(image_path, hue_change=0, saturation_change=0, value_change=0, display_result=True):
    """Read an image from disk, compute HSV-based features and return them.

    Parameters
    - image_path (str): path to the image file
    - hue_change (int): unused currently, intended hue delta (0-179)
    - saturation_change (int): unused currently, intended saturation delta
    - value_change (int): unused currently, intended value/brightness delta
    - display_result (bool): unused currently

    Returns
    - numpy array of all parameters
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Error, the image path is not {image_path}")

    img_small = cv2.resize(original_img, (100, 100), interpolation=cv2.INTER_AREA)

    # Convert the image from BGR (OpenCV default) to HSV
    hsv_img = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    # KMeans clustering on pixels to find dominant HSV colors
    pixels = hsv_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    flat_dominant_colors = dominant_colors.flatten()

    # Hue channel and warm mask
    h_channel = hsv_img[..., 0]
    warm_mask = (h_channel < 30) | (h_channel > 150)
    warm_pixel_count = np.sum(warm_mask)
    total_pixels = h_channel.size
    warm_cool_ratio = warm_pixel_count / total_pixels

    # Saturation and value stats
    s_channel = hsv_img[..., 1]
    avg_saturation = np.mean(s_channel) / 255.0
    saturation_contrast = np.std(s_channel) / 255.0

    v_channel = hsv_img[..., 2]
    avg_brightness = np.mean(v_channel) / 255.0
    brightness_contrast = np.std(v_channel) / 255.0


    stats_vec = np.array([warm_cool_ratio, avg_brightness,saturation_contrast, 
                          avg_brightness, brightness_contrast])
    final_vec = np.concatenate([flat_dominant_colors, stats_vec])

    return final_vec

# test in this program

if __name__ == "__main__":
    '''
    img_path = '/Users/rzzz/Desktop/Workspace/Projects/recommendation/Screenshot 2025-12-12 at 14.46.28 (2).png'
    hue_change=4
    saturation_change = 10
    value_change = 3

    modified_image = modify_hsv_img(img_path,hue_change,saturation_change,value_change)
    '''
    pass
    
