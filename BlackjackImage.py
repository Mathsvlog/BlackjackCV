import cv2
from math import cos,sin,pi,acos
import numpy as np
import PointFunctions as pt
from BlackjackGlobals import *

class BlackjackImage:

	def __init__(self, image, drawImageOut=True):
		self.image = image
		self.imageGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		self.drawOut = drawImageOut
		if drawImageOut:
			self.imageOut = np.copy(image)
		# centroid approximation
		epsilon = lambda c:0.07*cv2.arcLength(c, True)
		self.approx = lambda c:cv2.approxPolyDP(c, epsilon(c), True)
		# 1D numpy array to tuple
		self.toPoints = lambda points:map(lambda p:tuple(p[0]), points)

	def getInputImage(self):
		return self.image

	def getOutputImage(self):
		return self.imageOut

	def show(self, name="BlackjackCV", width=900):
		im = self.imageOut if self.drawOut else self.image
		w,h = len(im[0]),len(im)
		scale = float(width)/w
		imageScaled = cv2.resize(im, (int(w*scale),int(h*scale)))
		cv2.imshow(name, imageScaled)

	"""
	Returns a list of centroids of corners using Harris corner algorithm
	"""
	def extractHarisCorners(self):
		#imValues = cv2.cornerHarris(self.imageGrey, 5, 3, 0.04)
		#imValues = cv2.cornerHarris(self.imageGrey, 8, 9, 0.04)
		imValues = cv2.cornerHarris(self.imageGrey, 5, 9, 0.04)
		ret, imCorners = cv2.threshold(imValues, 0.04*imValues.max(), 255, 0)
		imCorners = np.uint8(imCorners)
		_, _, _, centroids = cv2.connectedComponentsWithStats(imCorners)

		# handle situation where center pixel is wrongly considered an edge
		c = centroids[0]
		if abs(c[0]-imageX/2)<3 and abs(c[1]-imageY/2)<3 and imCorners[c[1],c[0]]==0:
			return centroids[1:]

		return centroids

	"""
	Draw circles at corner centroids
	"""
	def drawCorners(self, cornerCentroids, color=(0,255,0)):
		if self.drawOut:
			for x,y in cornerCentroids:
				x,y = int(x),int(y)
				cv2.circle(self.imageOut, (x,y), 2, color, -1)

	"""
	Returns a list of contours using Canny edge detection
	"""
	def extractCannyContours(self):
		canny = cv2.Canny(self.image, 100, 200)
		_, contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		return contours

	"""
	Draw edge contours
	"""
	def drawContours(self, contours, color):
		if isinstance(contours, np.ndarray):
			contours = [contours,]
		if self.drawOut:
			for c in contours:
				cv2.drawContours(self.imageOut, [c],0,color,1)


	"""
	Uses edges and corners to find potenial cards in the image
	"""
	def extractCardCandidates(self):
		contours = self.extractCannyContours()
		contourApprox = map(lambda c:self.approx(c), contours)
		self.drawContours(contours, (255,255,0))
		corners = self.extractHarisCorners()
		cornerList = map(lambda c:tuple(c),list(corners))

		condition = lambda i:len(contourApprox[i])==4 and 800<cv2.contourArea(contours[i])
		cardCandidates = []
		for idx in filter(condition, range(len(contours))):
			contour,appr = contours[idx], contourApprox[idx]
			points = self.toPoints(appr)
			
			isCard = True
			prevCornerList = cornerList[:]
			# contour is card if matchs 4 corners
			for i in xrange(4):
				point = points[i]
				closestDist = float("inf")
				closest = None
				for corner in cornerList:
					currDist = pt.dist(corner,point)
					if currDist < closestDist:
						closestDist = currDist
						closest = corner
				if closestDist > 5:
					isCard = False
					break
				else:
					appr[i] = closest
					cornerList.remove(closest)

			if isCard:
				self.drawContours(appr, (0,0,255))
				cardCandidates.append(self.toPoints(appr))

			else:
				prevCornerList = cornerList[:]

		self.drawCorners(corners)
		return cardCandidates
