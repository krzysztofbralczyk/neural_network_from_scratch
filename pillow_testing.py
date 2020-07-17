from __future__ import print_function
from PIL import Image
import numpy as np

im = Image.open("./assets/cat.jpg")
im_arr = np.array(im)
print(type(im_arr))
print(im_arr)

im2 = Image.fromarray(im_arr)
im2.show()
# print(im.format, im.size, im.mode)
# im.show()
# im.thumbnail((500, 500))
# r, g, b = im.split()
# print(r)
# r.show()
# g.show()
# b.show()
# merged_im = Image.merge("RGB", (r, g, b))

# merged_im.show()
# pixels = list(r.getdata())
# print(pixels)
