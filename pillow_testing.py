from __future__ import print_function
from PIL import Image

im = Image.open("./assets/cat.jpg")
print(im.format, im.size, im.mode)
im.show()