import numpy as np 
import cv2
from matplotlib import pyplot as plt 
from icecream import ic 

"https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/"

def drawlines(img1src, img2src, lines, pts1src, pts2src):
	''' img1 - image on which we draw the epilines for the points in img2
	lines - corresponding epilines '''

	r, c = img1src.shape
	img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
	img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
	# Edit: use the same random seed so that two images are comparable!
	np.random.seed(0)
	for r, pt1, pt2 in zip(lines, pts1src, pts2src):
		color = tuple(np.random.randint(0, 255, 3).tolist())
		x0, y0 = map(int, [0, -r[2]/r[1]])
		x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
		img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
		img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
		img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
	return img1color, img2color


class DM_Estimator():
	def __init__(self, img_name):
		self.img = img_name,
		self.left_kp = None,
		self.right_kp = None,
		self.left_des = None,
		self.right_des = None,
		self.matches = None,
		self.matchesMask = None,
		self.good = None,
		self.pts1 = None,
		self.pts2 = None,
		self.fundamental_matrix = None,
		self.inliers = None

	def detectKeyPoints(self, L, R, show = False):
		'''
		Inputs : 
			- Left img
			- Right img

		Returns : 
			- Keypoints left 
			- keypoints right 
			- Descriptors left 
			- Descriptors right 
		'''

		sift = cv2.SIFT_create()

		self.left_kp, self.left_des = sift.detectAndCompute(L, None)
		self.right_kp, self.right_des = sift.detectAndCompute(R, None)

		if show:
			self.showKP(L, R)


	def showKP(self, L, R):
		left_imgSIFT = cv2.drawKeypoints(L, self.left_kp, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		right_imgSIFT = cv2.drawKeypoints(R, self.right_kp, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		img_SIFT = np.hstack((left_imgSIFT, right_imgSIFT))

		cv2.imshow("SIFT", img_SIFT)
		cv2.waitKey()


	def matchKP(self):
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)   # or pass empty dictionary
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		self.matches = flann.knnMatch(self.left_des, self.right_des, k=2)

		self.matchesMask = [[0, 0] for i in range(len(self.matches))]
		self.good = []
		self.pts1 = []
		self.pts2 = []

		for i, (m, n) in enumerate(self.matches):
			if m.distance < 0.7*n.distance:	
				# Keep this keypoint pair
				self.matchesMask[i] = [1, 0]
				self.good.append(m)
				self.pts2.append(self.right_kp[m.trainIdx].pt)
				self.pts1.append(self.left_kp[m.queryIdx].pt)

	def showMatches(self, L, R):
		draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=self.matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)

		keypoint_matches = cv2.drawMatchesKnn(L, self.left_kp, R, self.right_kp, self.matches, None, **draw_params)
		cv2.imshow("Keypoint matches", keypoint_matches)
		cv2.waitKey()


	def computeFundamentalMat(self):

		# ic(self.pts1, self.pts2)
		self.pts1 = np.int32(self.pts1)
		self.pts2 = np.int32(self.pts2)

		self.fundamental_matrix, self.inliers = cv2.findFundamentalMat(self.pts1, self.pts2, cv2.FM_RANSAC)

		self.pts1 = self.pts1[self.inliers.ravel() == 1]
		self.pts2 = self.pts2[self.inliers.ravel() == 1]

	def epiLines(self, L, R):		

		# Find epilines corresponding to points in right image (second image) and
		# drawing its lines on left image
		# ic(self.fundamental_matrix, self.inliers)

		self.lines1 = cv2.computeCorrespondEpilines(np.array(self.pts2).reshape(-1, 1, 2), 2, self.fundamental_matrix)
		self.lines1 = self.lines1.reshape(-1, 3)
		img5, _ = drawlines(L, R, self.lines1, self.pts1, self.pts2)

		# Find epilines corresponding to points in left image (first image) and
		# drawing its lines on right image
		self.lines2 = cv2.computeCorrespondEpilines(np.array(self.pts1).reshape(-1, 1, 2), 1, self.fundamental_matrix)
		self.lines2 = self.lines2.reshape(-1, 3)
		img3, _ = drawlines(R, L, self.lines2, self.pts2, self.pts1)

		# plt.subplot(121), plt.imshow(img5)
		# plt.subplot(122), plt.imshow(img3)
		# plt.suptitle("Epilines in both images")
		# plt.show()

	def rectifieImgs(self, L, R):
		h1, w1 = L.shape
		h2, w2 = R.shape

		_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(self.pts1), np.float32(self.pts2), self.fundamental_matrix, imgSize=(w1, h1))
		
		L_rectified = cv2.warpPerspective(L, H1, (w1, h1))
		R_rectified = cv2.warpPerspective(R, H2, (w2, h2))

		fig, axes = plt.subplots(1, 2, figsize=(15, 10))
		axes[0].imshow(L_rectified, cmap="gray")
		axes[1].imshow(R_rectified, cmap="gray")
		axes[0].axhline(250)
		axes[1].axhline(250)
		axes[0].axhline(450)
		axes[1].axhline(450)
		plt.suptitle("Rectified images")
		# plt.savefig("rectified_images.png")
		# plt.show()

		return L_rectified, R_rectified

	def computeDepthEstimation(self, L_rectified, R_rectified):
		'''
		- Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
		- num_disp : Maximum disparity minus minimum disparity. The value is always greater than zero.
			In the current implementation, this parameter must be divisible by 16.	

		- uniquenessRatio : Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
			Normally, a value within the 5-15 range is good enough
		
		- speckleWindowsSize : Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
			Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
		
		- speckleRange : Maximum disparity variation within each connected component.
			If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
			Normally, 1 or 2 is good enough.
		'''
		block_size = 5
		min_disp = 0
		max_disp = 128

		num_disp = max_disp - min_disp

		uniquenessRatio = 5
		speckleWindowSize = 200

		speckleRange = 2
		disp12MaxDiff = 0

		stereo = cv2.StereoSGBM_create(
			minDisparity = min_disp,
			numDisparities = num_disp,
			blockSize = block_size,
			uniquenessRatio = uniquenessRatio,
			speckleWindowSize = speckleWindowSize,
			speckleRange = speckleRange,
			disp12MaxDiff = disp12MaxDiff,
			P1 = 8 * 1 * block_size * block_size,
			P2 = 32 * 1 * block_size * block_size,
		)

		disparity_SGBM = stereo.compute(L_rectified, R_rectified)
		disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
		disparity_SGBM = np.uint8(disparity_SGBM)
		# cv2.imshow("Disparity", disparity_SGBM)	
		# cv2.waitKey()

		return disparity_SGBM, block_size

