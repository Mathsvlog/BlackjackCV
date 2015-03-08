import cv2
from math import cos,sin,pi,acos,atan2,atan
import numpy as np
import PointFunctions as pf
from BlackjackGlobals import *
import itertools

class BlackjackImage:
	_projectionTransform = None
	_projectedCardSize = None
	_cardSizeTolerance = 7
	# tolerance of at least 6

	"""
	Class for an input image to analyze for playing cards
	"""
	def __init__(self, image, drawImageOut=True, project=False, recomputeProjection=False):
		self.image = image
		self.isProjected = False
		
		# 1D numpy array to tuple
		self.toPoints = lambda points:map(lambda p:tuple(p[0]), points)
		# centroid approximation
		epsilon = lambda c:0.07*cv2.arcLength(c, True)
		self.approx = lambda c:cv2.approxPolyDP(c, epsilon(c), True)
		self.drawOut = drawImageOut
		self._createImageVariants()

		# Perform a perspective transformation
		if project:
			if recomputeProjection or BlackjackImage._projectionTransform is None:
				self._computeProjectionTransform()
			if not BlackjackImage._projectionTransform is None:
				self._applyProjectionTransform()

		self.isProjected = project and BlackjackImage._projectedCardSize!=None


	"""
	Finds pixels that have a darker color than most of the image and darkens them further
	thresholdFactor in range 0 to 1, 0 uses average as threshold, 1 uses 255 as threshold
	darkenFactor is multiplied by the rgb values of dark pixels
	"""
	def _darkenImage(self, image, thresholdFactor, darkenFactor):
		image = np.copy(image)
		grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		mean = np.mean(grey)
		mean = mean + (255-mean)*thresholdFactor
		if darkenFactor>0:
			image[grey<mean] /= darkenFactor
		else:
			image[grey<mean] = 0
		return image

	"""
	Computes a perspective transform matrix to make cards perfectly rectangular and uniform size.
	Uses a card candidate from the image to compute the transform
	"""
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
		corners = [np.transpose(np.matrix([size[1], 0, 1])), np.transpose(np.matrix([size[1], size[0], 1]))]
		p = M*corners[0]
		angle = -atan2(p[1,0],p[0,0])
		M = np.matrix([[cos(angle), -sin(angle),0],[sin(angle), cos(angle),0],[0,0,1]])*M
		# fix rotation to the z axis
		#M[1,0] = 0
		#M[2,0] = 0
		# scale to fix corners
		scale = float("inf")
		for i,j in [[0,1],[1,0]]:
			p = M*corners[i]
			scale = min(scale, size[j]*p[2,0]/p[i,0])
		M = np.matrix([[scale,0,0],[0,scale, 0],[0,0,1]])*M
		#print "\n".join(map(lambda m:str(m),M.tolist()))
		# set final matrix
		BlackjackImage._projectionTransform = M
		# compute card lengths in projected image
		pts = map(lambda c:zip(*(c/c[2]).tolist()[:2]), map(lambda c:M*np.matrix([[c[0]],[c[1]],[1]]), card))
		cardLengths = map(lambda i:pf.dist(pts[0][0],pts[i][0]), range(1,4))
		cardLengths.sort()
		BlackjackImage._projectedCardSize = tuple(map(lambda l:int(round(l)), cardLengths))

	def _applyProjectionTransform(self):
		# project original image
		rotated = cv2.warpPerspective(self.image, BlackjackImage._projectionTransform, (640,480))
		self.image = np.copy(rotated)
		self._createImageVariants()
	
	def _createImageVariants(self):
		if self.drawOut:
			self.imageOut = np.copy(self.image)
		self.imageDark = self._darkenImage(self.image, .5, 0)# darken
		self.imageGrey = cv2.cvtColor(self.imageDark, cv2.COLOR_BGR2GRAY)


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
		if self.isProjected:
			cv2.imshow("grey", self.imageGrey)
			block = min(BlackjackImage._projectedCardSize)/6
			imValues = cv2.cornerHarris(self.imageGrey, block, 9, 0.04)
		else:
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
		canny = cv2.Canny(self.imageDark, 100, 200)
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
		# find contours
		contours = self.extractCannyContours()
		contourApprox = map(lambda c:self.approx(c), contours)
		self.drawContours(contours, (255,255,0))

		# whiten contours before finding harris corners
		if self.isProjected:
			for _ in range(2):
				for c in contours:
					cv2.drawContours(self.imageGrey, [c], 0, (255,255,255), -1)
					cv2.drawContours(self.imageDark, [c], 0, (255,255,255), -1)
				contours = self.extractCannyContours()
				contourApprox = map(lambda c:self.approx(c), contours)


		# find corners
		corners = self.extractHarisCorners()
		cornerList = map(lambda c:tuple(c),list(corners))
		self.drawCorners(corners)

		if self.isProjected:
			cardCandidates = self._extractCardCandidatesHelper2(contours, contourApprox, cornerList)
			for c in cardCandidates:
				self.drawCorners(c, (255,0,0))
		else:
			cardCandidates = self._extractCardCandidatesHelper1(contours, contourApprox, cornerList)
		
		if self.isProjected:
			for pts in cardCandidates:
				cardLengths = map(lambda i:pf.dist(pts[0],pts[i]), range(1,4))
				cardLengths.sort()
				errors = map(lambda i:abs(cardLengths[i]-BlackjackImage._projectedCardSize[i]), range(3))
				#print max(errors)
		return cardCandidates

	def _extractCardCandidatesHelper2(self, contours, contourApprox, cornerList):
		corners = cornerList[:][:]
		candidates = []
		# match first corner in list until list is depleted
		while len(corners) > 1:
			c1 = corners[0]
			#cv2.circle(self.imageOut, tuple(map(lambda i:int(i),c1)), 10, (255,255,0), -1)
			# compute distances from all other corners to first corner
			dists = map(lambda i:pf.dist(c1, corners[i]), range(1,len(corners)))
			# find which corner are the proper distances to be part of the same card
			neighbors = {i:[] for i in [0,1,2]}
			for d in range(len(dists)):
				for i in range(3):
					if abs(dists[d] - BlackjackImage._projectedCardSize[i]) < BlackjackImage._cardSizeTolerance:
						neighbors[i].append(corners[d+1])
						#color = [0,0,0]; color[i]=255
						#cv2.circle(self.imageOut, tuple(map(lambda i:int(i),corners[d+1])), 5, color, -1)#
			foundCard = False
			for i in [2,1,0]:
				for c2 in neighbors[i]:
					match, candidate, (i,j) = self._cornerMatch(c1, c2, i)
					if match:
						c3,c4 = candidate[i], candidate[j]
						for c in corners[1:]:
							if pf.dist(c3,c) < BlackjackImage._cardSizeTolerance:
								candidate[i] = c3
								corners.remove(c)
								break
						for c in corners[1:]:
							if pf.dist(c4,c) < BlackjackImage._cardSizeTolerance:
								candidate[j] = c4
								corners.remove(c)
								break
						corners.remove(c1)
						corners.remove(c2)
						candidates.append(candidate)
						foundCard = True
						break
				if foundCard:
					#print candidates[-1]
					#print len(corners)
					break
			if not foundCard:
				corners = corners[1:]

		candidates = map(lambda cand:map(lambda c:(int(round(c[0])),int(round(c[1]))), cand), candidates)
		return candidates

	def _cornerMatch(self, c1, c2, i):
		def isWhite(candidate):
			rect = np.matrix(map(lambda c:(int(round(c[0])),int(round(c[1]))), candidate))
			mask = np.zeros((imageY,imageX), np.uint8)
			cv2.drawContours(mask, [rect],0,255,-1)
			pts = np.nonzero(mask)
			whiteness = np.mean(self.imageGrey[pts])
			return whiteness > 250

		# short side
		if i==0:
			l = 1.4
			for a,b in [(c1,c2),(c2,c1)]:
				c3 = pf.lerp(c2,pf.add(c2,pf.norm(a,b)),l)
				c4 = pf.lerp(c1,pf.add(c1,pf.norm(a,b)),l)
				candidate = [c1,c2,c3,c4]
				if isWhite(candidate):
					return True, candidate, (2,3)
		# long side
		elif i==1:
			l = 1.4
			for a,b in [(c1,c2),(c2,c1)]:
				c3 = pf.lerp(c2,pf.add(c2,pf.norm(a,b)),l)
				c4 = pf.lerp(c1,pf.add(c1,pf.norm(a,b)),l)
				candidate = [c1,c2,c3,c4]
				if isWhite(candidate):
					return True, candidate, (2,3)
		# diagonal
		elif i==2:
			for a,l in ((atan(2.5/3.5), 3.5/(2.5**2+3.5**2)**.5),(atan(3.5/2.5), 2.5/(2.5**2+3.5**2)**.5)):
				c,s = cos(a),sin(a)
				p = pf.vec(c1,c2)
				c3 = pf.lerp(c1,pf.add(c1,(c*p[0]-s*p[1],s*p[0]+c*p[1])),l)
				p = pf.vec(c2,c1)
				c4 = pf.lerp(c2, pf.add(c2,(c*p[0]-s*p[1],s*p[0]+c*p[1])), l)
				if pf.dist(c1,c3) > pf.dist(c1,c4):
					c3,c4 = c4,c3
				candidate = [c1,c3,c2,c4]
				if isWhite(candidate):
					return True, candidate, (1,3)

		return False, None, (-1,-1)


	def _extractCardCandidatesHelper1(self, contours, contourApprox, cornerList):
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
		return cardCandidates