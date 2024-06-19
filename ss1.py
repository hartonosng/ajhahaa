import base64
import io
from PIL import Image
import torch

def load_image_from_base64(base64_string, input_size=448, max_num=6):
    # Fix incorrect padding
    padding_needed = 4 - (len(base64_string) % 4)
    if padding_needed:
        base64_string += "=" * padding_needed
    
    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_string)
    
    # Convert the bytes data to a PIL image
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Define your transformation and preprocessing functions
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    
    # Apply the transformation to each image
    pixel_values = [transform(image) for image in images]
    
    # Stack the pixel values into a single tensor
    pixel_values = torch.stack(pixel_values)
    
    return pixel_values

# Example usage:
# base64_string = "your_base64_encoded_image_string_here"
# pixel_values = load_image_from_base64(base64_string)
