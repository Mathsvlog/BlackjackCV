from BlackjackGlobals import cardSize
import cv2
import numpy as np
from time import time

class BlackjackComparer:

	def __init__(self):
		# ORB used for feature detection
		self.orb = cv2.ORB_create()
		self.orb.setFastThreshold(5)
		self.orb.setEdgeThreshold(3)
		#self.orb.setMaxFeatures(200)

		self.bf = cv2.BFMatcher()
		# histogram grid size
		self.gridX, self.gridY = 7,7
		# read 52 input cards and extract features
		self._computeTrainSet()

	def _computeTrainSet(self):
		train = {}
		for v in "A23456789TJQK":
			for s in "DCHS":
				name = v+s+".jpg"
				f = "cards/"+name
				im = self._getCard(f)
				im = cv2.resize(im, cardSize)
				train[v+s] = im
		self.train = train
		# tuple containing (card, thresholded card, feature histogram, descriptors)
		self.trainData = {k: self.computeData(train[k]) for k in train.keys()}

	def _getCard(self, filename):
		im = cv2.imread(filename)
		#im = cv2.resize(im, cardSize)
		return im

	def _compare(self, data1, data2):
		c1, h1, d1 = data1
		c2, h2, d2 = data2
		h3 = {k: abs(h1[k] - h2[k]) for k in h1.keys()}
		hMetric = sum(h3.values())/len(h3.values())
		#matches = self.bf.match(d1, d2)
		#dMetric = sum(map(lambda m:m.distance, matches))/len(matches)

		th1 = self._thresholdCard(c1)
		th2 = self._thresholdCard(c2)
		im = cv2.absdiff(th1, th2)
		tMetric = sum(cv2.mean(im))/max(sum(cv2.mean(th1)),sum(cv2.mean(th2)))

		return hMetric*tMetric

	def _thresholdCard(self, card):
		return cv2.threshold(card, 128, 255, cv2.THRESH_BINARY)[1]

	def _compare2(self, data1, data2):
		c1, th1, h1, d1 = data1
		c2, th2, h2, d2 = data2
		im = cv2.absdiff(th1, th2)
		tMetric = sum(cv2.mean(im))/max(sum(cv2.mean(th1)),sum(cv2.mean(th2)))
		h3 = {k: abs(h1[k] - h2[k]) for k in h1.keys()}
		hMetric = sum(h3.values())/len(h3.values())
		return tMetric*hMetric

	def computeData(self, im, draw=False):
		nx,ny = self.gridX, self.gridY
		
		keys,desc = self.orb.detectAndCompute(im, None)

		hist = {(i,j):0 for i in range(nx) for j in range(ny)}
		h,w,_ = np.shape(im)
		# draw grid lines
		if draw:
			for i in xrange(1,nx):
				cv2.line(im, (w*i/nx, 0), (w*i/nx, h), (255,100,100), 3)
			for i in xrange(1,ny):
				cv2.line(im, (0, h*i/ny), (w, h*i/ny), (255,100,100), 3)
		# compute distribution of keypoints in grid
		for k in keys:
			x,y = k.pt
			x,y = int(x),int(y)
			if draw:
				cv2.circle(im, (x,y), 4, (0,255,0), -1)
			i,j = (w-x)*nx/w, (h-y)*ny/h
			#if i!=0 and i!=nx-1:
			hist[(i, j)] += k.size

		# normalize histogram
		total = float(sum(hist.values()))
		if total!=0:
			hist = {k:hist[k]/total for k in hist.keys()}

		th = self._thresholdCard(im)
		return (im, th, hist, desc)


	def getClosestCard(self, card, draw=False):
		h1 = self.computeData(card, draw)
		vals = {k: self._compare2(h1,self.trainData[k]) for k in self.trainData.keys()}
		match = min(vals, key=vals.get)
		return match, vals[match]

	def getClosestCards(self, card, numCards=1):
		# create score for each card
		data1 = self.computeData(card, False)
		vals1 = {k: self._compare2(data1,self.trainData[k]) for k in self.trainData.keys()}
		# flip card and recompute scores
		c,th,hist,d=data1
		c = c[::-1,::-1]
		th = self._thresholdCard(c)
		hist = {(i,j):hist[(self.gridX-i-1, self.gridX-j-1)] for i,j in hist.keys()}
		data2 = (c,th,hist,d)
		vals2 = {k: self._compare2(data2,self.trainData[k]) for k in self.trainData.keys()}
		# pick best score for each card
		vals = {k:min(vals1[k],vals2[k]) for k in vals1.keys()}
		# sort by score and return number of cards requested
		vals = sorted(vals.items(), key=lambda v:v[1])
		return zip(*vals[:numCards])

	def _test(self):
		print "START"
		t = time()
		for s in "DCHS":
			for v in "A23456789TJQK":
				name = v+s+".jpg"
				c = cv2.imread("train/original/"+name)
				c = cv2.resize(c, cardSize)
				c = cv2.blur(c, (30,30))
				#match, val = self.getClosestCard(c)
				matches, values = self.getClosestCards(c, 15)
				if matches[0] != v+s:
					if v+s in matches:
						print v+s, matches.index(v+s)
					else:
						print v+s, matches
		print "END", time()-t


#b = BlackjackComparer()
#b._test()