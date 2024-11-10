import os
import cv2

# (VIDEO) CREATE A VIDEO FROM ALL THE FRAMES
fourcc = cv2.VideoWriter_fourcc(*'XVID') # FourCC is a 4-byte code used to specify the video codec

for i in range(9999999999999):
    # Get the number of images and then figure out what is the latest image. Hence with
    # this image we are going to start with and let it dream on and on
    if os.path.isfile('data/img_{}.jpg'.format(i+1)):
        pass
    # Figure out how long the dream is
    else:
        dream_length = i
        break
print(f"total frames in this video: {dream_length + 1}")

# Specify the fourCC, frames per second (fps), and frame rate
# The frames per second value depends on:
# 1. The number of frames we have created. The lower the number of frames, the lower the fps.
# 2. For example, 1080 pixel image 60 fps. The bigger the image the higher the fps.
out = cv2.VideoWriter('data/deepdreamvideo.avi', fourcc , 5.0, (910, 605))

for i in range(dream_length):
    # Build the frames of cv2.VideoWriter
    frame = cv2.imread('data/img_{}.jpg'.format(i))
    out.write(frame)

out.release()
print("Done!")