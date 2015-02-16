from BlackjackGlobals import cardSize
import cv2
import numpy as np

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

		_, th1 = cv2.threshold(c1, 128, 255, cv2.THRESH_BINARY)
		_, th2 = cv2.threshold(c2, 128, 255, cv2.THRESH_BINARY)
		im = cv2.absdiff(th1, th2)
		tMetric = sum(cv2.mean(im))/max(sum(cv2.mean(th1)),sum(cv2.mean(th2)))

		return hMetric*tMetric

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

		_,th = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
		return (im, th, hist, desc)


	def getClosestCard(self, card, draw=False):
		h1 = self.computeData(card, draw)
		vals = {k: self._compare2(h1,self.trainData[k]) for k in self.trainData.keys()}
		match = min(vals, key=vals.get)
		return match, vals[match]

	def getClosestCards(self, card, numCards=1):
		data1 = self.computeData(card, False)
		vals1 = {k: self._compare2(data1,self.trainData[k]) for k in self.trainData.keys()}
		
		card = card[::-1,::-1]
		data = self.computeData(card, False)
		vals2 = {k: self._compare2(data,self.trainData[k]) for k in self.trainData.keys()}

		"""
		c,th,hist,d=data1
		hist = {(i,j):hist[(self.gridX-i-1, self.gridX-j-1)] for i,j in hist.keys()}
		data2 = (c,th,hist,d)
		vals2 = {k: self._compare2(data2,self.trainData[k]) for k in self.trainData.keys()}
		"""
		vals = {k:min(vals1[k],vals2[k]) for k in vals1.keys()}
		

		vals = sorted(vals.items(), key=lambda v:v[1])
		return zip(*vals[:numCards])

	def _test(self):
		print "START"
		for s in "DCHS":
			for v in "A23456789TJQK":
				name = v+s+".jpg"
				c = cv2.imread("train/original/"+name)
				c = cv2.blur(c, (30,30))
				#match, val = self.getClosestCard(c)
				matches, values = self.getClosestCards(c, 15)
				if matches[0] != v+s:
					if v+s in matches:
						print v+s, matches.index(v+s)
					else:
						print v+s, matches
		print "END"


#b = BlackjackComparer()
#b._test()



if False:
	canny = lambda s: cv2.Canny(s, 100, 200)
	names = ["5H", "4H", "JS", "JC"]
	sharp = map(lambda n:cv2.imread("train/sharp/"+n+".jpg"), names)
	orig = map(lambda n:cv2.imread("train/original/"+n+".jpg"), names)
	sharpCanny = map(canny, sharp)

	c = canny(orig[2])
	for i in range(len(names)):
		for j in range(len(names)):

			#im = cv2.absdiff(sharpCanny[i], c)
			#cv2.imshow(names[i], sharp[i])
			_, th1 = cv2.threshold(sharp[i], 128, 255, cv2.THRESH_BINARY)
			_, th2 = cv2.threshold(orig[j], 128, 255, cv2.THRESH_BINARY)
			im = cv2.absdiff(th1, th2)

			#print sum(cv2.mean(im))/255, sum(cv2.mean(th1))/255
			print sum(cv2.mean(im))/sum(cv2.mean(th1))
			cv2.imshow(names[i], im)
		print

	cv2.waitKey(0)


if False:
	orb = cv2.ORB_create()
	orb.setFastThreshold(5)
	orb.setEdgeThreshold(3)
	brisk = cv2.BRISK_create()

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	card = cv2.imread("cards/3D.jpg")
	im = cv2.imread("images/17.jpg")
	cv2.matchTemplate(im, card, cv2.TM_SQDIFF_NORMED)

	k1,d1 = brisk.detectAndCompute(card, None)
	k2,d2 = brisk.detectAndCompute(im, None)

	matches = bf.knnMatch(d1, d2, k=1)
	matches = map(lambda m:m[0], filter(lambda m:len(m)>0, matches))
	matches = sorted(matches, key=lambda m:m.distance)
	out = cv2.drawMatches(card, k1, im, k2, matches[:30], 2)
	#desc = orb.detect(im)

	#matcher = cv2.DescriptorMatcher_create("BruteForce")
	#matcher.getTrainDescriptors(desc)
	def asdf(im, numFeatures=200):
		corners = cv2.goodFeaturesToTrack(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), numFeatures, 0.01, 10)
		print corners
		for c in corners:
			x,y = c[0]
			cv2.circle(im, (x,y), 2, (0,255,0), -1)


	cv2.imshow("image", im)
	cv2.imshow("card", card)
	cv2.imshow("out", out)

	cv2.waitKey(0)