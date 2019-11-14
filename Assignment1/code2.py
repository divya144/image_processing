from skimage.exposure import cumulative_distribution
import matplotlib.pylab as plt
import numpy as np
import cv2
def hist_matching(c, c_t, im):
 pixels = np.arange(256)
 new_pixels = np.interp(c, c_t, pixels) 
 im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
 return im

def gammatrans(img,gamma):
    rows,columns,_=img.shape
    # gamma=2
    img2=np.copy(img)
    for i in range(rows):
        for j in range(columns):
            value=img[i][j]
            value=np.clip(pow((value/255.0),gamma)*255.0,0,255)
            img2[i][j]=value
    return img2

img1=cv2.imread('image1.jpg')
plt.subplot(221)
plt.imshow(img1)
img2=cv2.imread('image2.jpg')
plt.subplot(222)
plt.imshow(img2)
c,_ = cumulative_distribution(img1) 
c_t,_ = cumulative_distribution(img2) 
img3=hist_matching(c,c_t,img1)
c_new,_=cumulative_distribution(img3)
plt.subplot(223)
plt.plot(c,'-b',label='image1')
plt.plot(c_t,label='image2')
plt.plot(c_new,label='new')
plt.subplot(224)
plt.imshow(img3)
plt.show()
plt.subplot(221)
plt.imshow(gammatrans(img3,1.5))
plt.subplot(222)
plt.imshow(gammatrans(img3,2))
plt.subplot(223)
plt.imshow(gammatrans(img3,0.8))
plt.subplot(224)
plt.imshow(gammatrans(img3,0.4))
plt.show()
