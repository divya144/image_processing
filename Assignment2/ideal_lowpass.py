###############IDEAL LOW PASS FILTER######################
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import numpy.fft as fp
import math

img=cv.imread('image3.png')
row,column,_=img.shape
black=[0,0,0]
plt.figure(figsize=(15,15))
# image= cv.copyMakeBorder(img,row/2,row/2,column/2,column/2,cv.BORDER_CONSTANT,value=black)
image= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.subplot(221),plt.imshow(image,cmap='gray')
P,Q=image.shape
#F(u,v)---fourier transform of image(x,y)
F=fp.fft2(image)
F=fp.fftshift(F)
# print("fourier transform",F)
H=np.zeros([P,Q])
print(H.shape)
D0=int(input("Enter cutoff frequency-"))
midr=P/2
midc=Q/2
for i in range(P):
    for j in range(Q):
        dist=math.sqrt(math.pow((i-midr),2)+math.pow((j-midc),2))
        if dist<=D0:
            H[i][j]=1
filter=np.clip(H,0,255)
plt.subplot(222),plt.imshow(filter,cmap='gray')
G = [ [ 0.0+0.0j for i in range(Q) ] for j in range(P) ] 
for i in range(P):
    for j in range(Q):
        G[i][j]=H[i][j]*F[i][j]

I5=np.log10(np.abs(G))
plt.subplot(223),plt.imshow(I5,cmap='gray')
# print("printing G",G)
g=fp.ifft2(fp.ifftshift(G)).real
print("printing g",g)
plt.subplot(224),plt.imshow(g,cmap='gray');plt.show()
        


