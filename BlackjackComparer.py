from BlackjackGlobals import cardSize
import cv2
import numpy as np

class BlackjackComparer:

	def __init__(self):
		self.orb = cv2.ORB_create()
		self.orb.setFastThreshold(10)
		self.orb.setEdgeThreshold(11)
		self.gridX, self.gridY = 12, 12
		self._computeTrainSet()

	def _computeTrainSet(self):
		train = {}
		for v in "A23456789TJQK":
			for s in "DCHS":
				name = v+s+".jpg"
				f = "cards/"+name
				im = self._getCard(f)
				train[v+s] = im
		self.train = train
		self.trainHist = {k: self.computeFeatureHistogram(train[k]) for k in train.keys()}

	def _getCard(self, filename):
		im = cv2.imread(filename)
		#im = cv2.resize(im, cardSize)
		return im

	def _compare(self, h1, h2):
		h3 = {k: abs(h1[k] - h2[k]) for k in h1.keys()}
		return sum(h3.values())/len(h3.values())

	def computeFeatureHistogram(self, im, draw=False):
		nx,ny = self.gridX, self.gridY#
		keys = self.orb.detect(im)
		hist = {(i,j):0 for i in range(nx) for j in range(ny)}
		h,w,_ = np.shape(im)

		for k in keys:
			x,y = k.pt
			x,y = int(x),int(y)
			if draw:
				cv2.circle(im, (x,y), 1, (0,255,0), -1)
			hist[((w-x)*nx/w, (h-y)*ny/h)] += 1
		total = float(sum(hist.values()))
		hist = {k:hist[k]/total for k in hist.keys()}

		# draw grid lines
		if draw:
			for i in xrange(1,nx):
				cv2.line(im, (w*i/nx, 0), (w*i/nx, h), (255,0,0))
			for i in xrange(1,ny):
				cv2.line(im, (0, h*i/ny), (w, h*i/ny), (255,0,0))

		return hist


	def getClosestCard(self, card):
		h1 = self.computeFeatureHistogram(card)
		vals = {k: self._compare(h1,self.trainHist[k]) for k in self.trainHist.keys()}
		match = min(vals, key=vals.get)
		return match, vals[match]

	def _test(self):
		print "START"
		for s in "DCHS":
			for v in "A23456789TJQK":
				name = v+s+".jpg"
				c = cv2.imread("train/original/"+name)
				c = cv2.blur(c, (10,10))
				match, val = self.getClosestCard(c)
				if match != v+s:
					print match, v+s
		print "END"
