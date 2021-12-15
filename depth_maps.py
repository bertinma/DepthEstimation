import numpy as np 
import cv2
from matplotlib import pyplot as plt 
import os 
from icecream import ic 



def showDisparity(left, right, bSize = 5):
	stereo = cv2.StereoBM_create(numDisparities = 32, blockSize=bSize)

	disparity = stereo.compute(left, right)

	d_min = disparity.min()
	d_max = disparity.max()

	return np.uint8(255*(disparity - d_min) / (d_max - d_min))



def show_inputs(left, right):
	plt.figure(figsize = (10, 5))
	plt.subplot(1, 2, 1)
	plt.imshow(left, "gray")
	plt.axis('off')
	plt.subplot(1,2,2)
	plt.imshow(right, "gray")
	plt.axis('off')
	plt.show()

if __name__ == "__main__":

	working_dir = r"./data/2018-10-31-06-55-01_"
	left_im = os.path.join(working_dir+"L", os.listdir(working_dir+"L")[0])
	right_im = os.path.join(working_dir+"R", os.listdir(working_dir+"R")[0])

	ic(left_im, right_im)

	if os.path.exists(left_im):
		L = cv2.imread(left_im, 0)
	if os.path.exists(right_im):
		R = cv2.imread(right_im, 0)


	result = showDisparity(L, R)
	plt.imshow(result, "gray")
	plt.axis("off")
	plt.show()

