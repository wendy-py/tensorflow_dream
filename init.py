import matplotlib.pyplot as plt
from PIL import Image

img_1 = Image.open("data/mars.jpg")
img_2 = Image.open('data/eiffel.jpg')
image = Image.blend(img_1, img_2, 0.5) # mix 50:50
plt.imshow(image)
plt.show()
image.save('data/img_0.jpg')
