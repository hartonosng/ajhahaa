import os
import shutil

# Define the source and destination folders
source_folder = 'output'
destination_folder = 'outputnew'

# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through all files in the source folder
for count, filename in enumerate(os.listdir(source_folder), start=1):
    if filename.endswith(".jpg"):  # Filter only for .jpg files
        source_path = os.path.join(source_folder, filename)
        new_filename = f"new_filename_{count}.jpg"
        destination_path = os.path.join(destination_folder, new_filename)
        
        # Move and rename the file
        shutil.move(source_path, destination_path)

print("Files moved and renamed successfully.")
