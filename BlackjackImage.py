import cv2
from math import cos,sin,pi,acos,atan2
import numpy as np
import PointFunctions as pf
from BlackjackGlobals import *

class BlackjackImage:
	_projectionTransform = None

	"""
	Class for an input image to analyze for playing cards
	"""
	def __init__(self, image, drawImageOut=True, project=False, recomputeProjection=False):
		self.image = image
		
		# 1D numpy array to tuple
		self.toPoints = lambda points:map(lambda p:tuple(p[0]), points)
		# centroid approximation
		epsilon = lambda c:0.07*cv2.arcLength(c, True)
		self.approx = lambda c:cv2.approxPolyDP(c, epsilon(c), True)
		self.drawOut = drawImageOut

			
		self.imageGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		if drawImageOut:
			self.imageOut = np.copy(image)

		# Perform a perspective transformation
		if project:
			if recomputeProjection or BlackjackImage._projectionTransform is None:
				self._computeProjectionTransform()
			if not BlackjackImage._projectionTransform is None:
				self._applyProjectionTransform()


	def _computeProjectionTransform(self):
		newImage = np.copy(self.image)
		candidates = self.extractCardCandidates()
		# Can't compute projection matrix if there are no candidates
		if len(candidates)==0:
			return

		# pick candidate closest to center
		size = np.shape(self.image)[:2]
		center = map(lambda x:x/2,size)
		candidateDists = map(lambda c:pf.dist(center, pf.avg(c)), candidates)
		card = candidates[np.argmin(candidateDists)]


		# reverse order if conunterclockwise
		if pf.ccw(card[0],card[1],card[2]):
			card[0],card[2] = card[2],card[0]

		# (idx-1, idx) forms the shortest edge
		idx = np.argmin(map(lambda i:pf.dist(card[i-1],card[i]), range(4)))
		box = card[:][:]

		# create square box
		for i,j in [(idx, idx-3), (idx-1, idx-2)]:
			box[j] = pf.rounded(pf.lerp(box[i], pf.add(box[i],pf.norm(box[idx],box[idx-1])), 1.4))

		# compute initial projection matrix
		M = cv2.getPerspectiveTransform(np.matrix(card,np.float32), np.matrix(box,np.float32))
		# translate to fix top left corner
		x,y = -M[0,2], -M[1,2]
		M = np.matrix([[1, 0,x],[0, 1,y],[0,0,1]])*M
		# rotate so top of image aligns
		corner = np.transpose(np.matrix([size[1], 0, 1]))
		p = M*corner
		angle = -atan2(p[1,0],p[0,0])
		M = np.matrix([[cos(angle), -sin(angle),0],[sin(angle), cos(angle),0],[0,0,1]])*M
		# scale to fix top right corner
		p = M*corner
		scale = size[1]*p[2,0]/p[0,0]
		M = np.matrix([[scale,0,0],[0,scale, 0],[0,0,1]])*M

		# set final matrix
		BlackjackImage._projectionTransform = M

	def _applyProjectionTransform(self):
		# project original image
		rotated = cv2.warpPerspective(self.image, BlackjackImage._projectionTransform, (640,480))
		self.imageOut = np.copy(rotated)
		self.image = np.copy(rotated)
		self.imageGrey = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

	def getInputImage(self):
		return self.image

	def getOutputImage(self):
		return self.imageOut

	"""
	Show image on a window
	"""
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
		#imValues = cv2.cornerHarris(self.imageGrey, 5, 9, 0.04)
		imValues = cv2.cornerHarris(self.imageGrey, 8, 9, 0.04)
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
	def drawContours(self, contours, color, thickness=1):
		if isinstance(contours, np.ndarray):
			contours = [contours,]
		if self.drawOut:
			for c in contours:
				cv2.drawContours(self.imageOut, [c],0,color,thickness)


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
					currDist = pf.dist(corner,point)
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
				self.drawContours(appr, (0,0,255), 2)
				cardCandidates.append(self.toPoints(appr))

			else:
				prevCornerList = cornerList[:]

		self.drawCorners(corners)
		return cardCandidates
