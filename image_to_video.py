import cv2
import os
 
# --- USER SETTINGS ---
image_folder = "ShockTest_Sec2Images"     # folder containing image filess
output_file = "ShockTest1_Video.mp4"
fps = 24               # frames per second
file_extension = ".png"     # image type -- if your images are jpg, png, etc.
codec = "mp4v"               # video codec ('mp4v', 'XVID', 'MJPG', etc.)
# ----------------------

# Get and sort image filenames from --- USER SETTINGS ---
images = []  
for f in os.listdir(image_folder):
    if f.endswith(file_extension):
        images.append(f)
images = sorted(images)

if not images:
    raise ValueError("No images found in the specified folder!")

# --- Inspect Contents ---
#print(images)
#exit()

# Read the first image to get frame size
first_frame_path = os.path.join(image_folder, images[0])
first_frame = cv2.imread(first_frame_path)
height, width, _ = first_frame.shape

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*codec)  # Codec for .mp4
video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

print(f"Creating video '{output_file}' from {len(images)} frames...")

# Loop through images and add them to the video
for i, filename in enumerate(images):
    filepath = os.path.join(image_folder, filename)
    frame = cv2.imread(filepath)

    if frame is None:
        print(f"⚠️ Skipping unreadable file: {filename}")
        continue

    video.write(frame)

video.release()
print("✅ Video creation complete!")