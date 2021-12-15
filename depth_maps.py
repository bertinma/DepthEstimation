import numpy as np 
import cv2
import os 
from icecream import ic 
from matplotlib import pyplot as plt 
from pathlib import Path 
from tqdm import tqdm 
import argparse 
from my_implementation import DM_Estimator 


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

	parser = argparse.ArgumentParser()
	parser.add_argument("-bs", "--block-size",  type = int, default = 5, help = "define block size", required = False)
	parser.add_argument("--my-impl", action="store_true", help= "select my implementation otherwise opencv implementation")
	parser.add_argument("-s", "--show", action="store_true", help= "select my implementation otherwise opencv implementation")
	opt = parser.parse_args()

	working_dir = r"./data/2018-10-31-06-55-01_"

	result_path = Path(f"./data/results_{opt.block_size}")
	result_path.mkdir(exist_ok = True)

	for left_im in tqdm(Path(working_dir+"L").glob("*.jpg")):
		right_im = Path(str(left_im).replace("L", "R"))

		if left_im.exists() and right_im.exists():

			L = cv2.imread(str(left_im), 0)
			R = cv2.imread(str(right_im), 0)

			if not opt.my_impl:

				result = showDisparity(L, R)
				plt.imsave(result_path.joinpath(left_im.name), result)


			else:
				depth_map_estimator = DM_Estimator(left_im.name)
				
				depth_map_estimator.detectKeyPoints(L, R, opt.show)

				depth_map_estimator.matchKP()
				if opt.show:
					depth_map_estimator.showMatches(L, R)

				depth_map_estimator.computeFundamentalMat()
				depth_map_estimator.epiLines(L, R)

				L_rect, R_rect = depth_map_estimator.rectifieImgs(L, R)

				disparity, bs = depth_map_estimator.computeDepthEstimation(L_rect, R_rect)
				result_path = Path(f"./data/results/my_impl_{bs}")
				result_path.mkdir(exist_ok = True)

				result = np.hstack((L, disparity))

				plt.imsave(result_path.joinpath(left_im.name), result)

				# break				

				result = None

