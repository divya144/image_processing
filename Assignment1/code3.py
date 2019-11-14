from skimage.exposure import cumulative_distribution
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy.fft as fp
img=cv2.imread('image2.jpg',0)
plt.imshow(img,cmap='gray');plt.show()
'Discrete Fourier Tranform'
dft=fp.fft2(img)
dft_shift = fp.fftshift(dft)
magnitude=np.log10(1+np.abs(dft_shift))
print(magnitude)
plt.subplot(221),plt.imshow(magnitude,cmap='gray')
phase=np.angle(dft_shift)
plt.subplot(222),plt.imshow(phase,cmap='gray')
#############Reconstruction##############
imr=fp.ifft2(np.abs(dft)).real
# plt.subplot(223),plt.imshow(imr,cmap='gray')
plt.subplot(223),plt.imshow(np.clip(imr,0,255),cmap='gray')
phase=np.exp(1j*np.angle(dft))
imp=fp.ifft2(phase).real
plt.subplot(224),plt.imshow(np.clip(imp,0,255),cmap='gray')
plt.show()

print(dft.shape)
minhf = int(input ("Enter minimum horizontal frequency coefficients index:"))
maxhf = int(input("Enter maximum horizontal frequency coefficients index:"))
minvf = int(input("Enter minimum vertical frequency coefficients index:"))
maxvf = int(input("Enter minimum vertical frequency coefficients index:"))
row,column=dft.shape
for i in range(row):
    for j in range(column):
        if((i<minhf or i>maxhf)and (j<minvf or j>maxvf)):
            dft[i][j]=0

print(dft[251][251],dft[300][300])
imr=fp.ifft2(dft).real
plt.subplot(221),plt.imshow(np.clip(imr,0,255),cmap='gray')
imr=fp.ifft2(np.abs(dft)).real
plt.subplot(222),plt.imshow(np.clip(imr,0,255),cmap='gray')
phase=np.exp(1j*np.angle(dft))
imp=fp.ifft2(phase).real
plt.subplot(223),plt.imshow(np.clip(imp,0,255),cmap='gray')
plt.show()