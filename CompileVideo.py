import cv2
import os

img_folder = "./CreatedImages"


images = [img for img in os.listdir(img_folder) if img.endswith(".jpg") or img.endswith(".png")]
images.sort()


if not images:
    raise ValueError("No images found in the folder")


frame = cv2.imread(os.path.join(img_folder, images[0]))
height, width, layers = frame.shape


fps = 60
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output_video_p5_4k.mp4', fourcc, fps, (width, height))


for image in images:
    img_path = os.path.join(img_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)


video.release()
print("Video created successfully!")
