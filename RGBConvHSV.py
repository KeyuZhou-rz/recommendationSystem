import cv2
import numpy as np
from sklearn.cluster import Kmeans

def modify_hsv_img(image_path,hue_change=0,saturation_change=0,value_change=0,display_result=True):
    original_img = cv2.imread(image_path)
    img_small = cv2.resize(original_img,(100,100),interpolation=cv2.INTER_AREA)
    if original_img is None:
        raise FileNotFoundError(f"Error, the image path is not {image_path}")
     # Convert the image from RGB to HSV
    hsv_img = cv2.cvtColor(img_small,cv2.COLOR_BGR2HSV)
    # Kmeans Logic applied here
    # turn (100,100,3) into (10000,3)
    pixels = hsv_img.reshape(-1,3)
    kmeans = Kmeans(n_clusters=5,random_state=42, n_init=10)
    kmeans.fit(pixels)

    # 5 dominant colors: Hue, Saturation, Value
    dominant_colors = kmeans.cluster_centers_

    # flatten them into a list of 15 numbers
    flat_dominant_colors = dominant_colors.flatten()

    # END of Kmeans
    # Apply hue change(wrap around for values outside 0-179), 
    # it selects every pixel's row, every pixel's column, but only the 0th channel (which is Hue in HSV).
    # h_channel = (hsv_img[...,0] + hue_change) % 180
    h_channel = hsv_img[...,0]
    '''
    Warm is defined as Red, Orange, Yellow
    In HSV, Red is 0-15, 165-179; Yellow is around 30.
    '''
    warm_mask = (h_channel<30) | (h_channel > 150)

    # Calculate ratio of warm pixels
    warm_pixel_count = np.sum(warm_mask)
    total_pixels = h_channel.size
    warm_cool_ratio = warm_pixel_count / total_pixels

    # Apply saturation and value changes with clipping(0-255)
    s_channel = hsv_img[...,1]
    # s_channel = np.clip(s_channel + saturation_change,0,255)
    # Average Saturation, and normalized to 0-1
    avg_saturation = np.mean(s_channel) / 255.0
    # Saturation Contrast
    saturation_contrast = np.std(s_channel) / 255.0
  
    # same way applied in v_channel
    v_channel = hsv_img[...,2]
    # v_channel = np.clip(hsv_img[...,2] + value_change,0,255)
    avg_brightness = np.mean(v_channel) / 255.0
    brightness_contrast = np.std(v_channel) / 255.0
    


    return [flat_dominant_colors, warm_cool_ratio, avg_saturation, saturation_contrast,
            avg_brightness, brightness_contrast]

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
    
