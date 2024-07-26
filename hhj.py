import cv2
import numpy as np

def resize_image(input_path, output_path, target_width, target_height):
    # Read the image using cv2
    img = cv2.imread(input_path)
    # Get original dimensions
    original_height, original_width = img.shape[:2]
    # Calculate the scaling factor to maintain the resolution
    scaling_factor = min(target_width / original_width, target_height / original_height)
    # Calculate the new dimensions
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    # Create a new blank image with the target dimensions
    final_img = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    # Calculate position to paste the resized image to center it
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    # Paste the resized image onto the blank image
    final_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
    # Save the image
    cv2.imwrite(output_path, final_img)

# Example usage
input_path = 'path/to/your/document_page.png'
output_path = 'path/to/save/resized_image.png'
target_width = 800  # Desired width
target_height = 1000  # Desired height

resize_image(input_path, output_path, target_width, target_height)
