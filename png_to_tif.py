from PIL import Image
import os

# Input and output folders
input_folder = "ValidationData/masks"
output_folder = "ValidationData/masks"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert all PNG files
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".tif"):
        png_path = os.path.join(input_folder, filename)

        # Create output filename
        tif_filename = os.path.splitext(filename)[0] + ".png"
        tif_path = os.path.join(output_folder, tif_filename)

        # Open and save as TIFF
        with Image.open(png_path) as img:
            img.save(tif_path, format="PNG")
        # with Image.open(png_path) as img:
        #     img.save(tif_path, format="TIFF")

        print(f"Converted: {filename} -> {tif_filename}")

print("Conversion complete.")