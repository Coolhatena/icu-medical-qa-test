import cv2
import numpy as np
from time import sleep

img_name = "img1.jpg"
q_unicode = ord('q') 

while True:
	src = cv2.imread("images/" + img_name)

	LOW = np.array([0, 0, 0])
	UPP = np.array([32, 255, 83])

	hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
	msk = cv2.inRange(hsv, LOW, UPP)
	filtered = cv2.bitwise_and(src,src, mask= msk)

	cv2.imshow('Frame', msk)

	kernel = np.ones((5, 5), np.uint8)
	img_dilation = cv2.dilate(msk, kernel, iterations=2)

	cv2.imshow('Dilated', img_dilation)

	key = cv2.waitKey(1)
	if key == q_unicode: # If 'q' is pressed, close program (Its case sensitive)
		break

cv2.destroyAllWindows()