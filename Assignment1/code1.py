from PIL import Image
import PIL
import cv2
im1=Image.open(r"image1.jpg")
level=int(input("enter number of level: "))
im1=im1.quantize(level)
im1.show()
