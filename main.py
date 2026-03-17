import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("StandardResolution.tiff")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#height, width = img_rgb.shape[:2]
#pooled_image = cv2.resize(img_rgb, (width // 4, height // 4), interpolation=cv2.INTER_AREA)

#crop = pooled_image[500:1000, 750:1500]


lower_rgb = np.array([160, 160, 0])   
upper_rgb = np.array([200, 200, 200])
mask_rgb = cv2.inRange(img_rgb, lower_rgb, upper_rgb)

mask_inverted = cv2.bitwise_not(mask_rgb)

kernel_open = np.ones((10,10), np.uint8)
mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_OPEN, kernel_open)
kernel_close = np.ones((12,12), np.uint8)
mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_CLOSE, kernel_close)

_, mask_inverted = cv2.threshold(mask_inverted, 1, 255, cv2.THRESH_BINARY)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_gray = np.array([0, 0, 50]) 
upper_gray = np.array([180, 50, 200])


edges = cv2.Canny(img_rgb, 50, 100)
mask = cv2.inRange(img_hsv, lower_gray, upper_gray)

plt.figure(figsize=(12,4))

#plt.subplot(1,3,1)
#plt.hist(crop[:,:,0].ravel(), bins=256)
#plt.title("Red channel")


#plt.subplot(1,3,2)
#plt.hist(crop[:,:,1].ravel(), bins=256)
#plt.title("Green channel")


#plt.subplot(1,3,3)
#plt.hist(crop[:,:,2].ravel(), bins=256)
#plt.title("Blue channel")

#plt.show()

plt.subplot(1, 2, 1)
plt.imshow(mask_inverted, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.show()
