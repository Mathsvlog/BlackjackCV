from BlackjackGlobals import cardSize,pipPartSize
import cv2
import numpy as np
from time import time
from BlackjackCard import BlackjackCard

class BlackjackComparer:

	"""
	Class for identifying the suit and value of playing card images
	"""
	def __init__(self):
		self._computeTrainSet()

	"""
	Computes the images that are used to test against all input cards
	"""
	def _computeTrainSet(self):
		self.pipSuits = {}
		self.pipValues = {}
		for s in "DCHS":
			f = "pips/"+s+".jpg"
			pip = cv2.imread(f)
			self.pipSuits[s] = self._threshold(cv2.resize(self._trim(pip), pipPartSize))
		for v in "A23456789TJQK":
			f = "pips/"+v+".jpg"
			pip = cv2.imread(f)
			self.pipValues[v] = self._threshold(cv2.cvtColor(cv2.resize(self._trim(pip), pipPartSize), cv2.COLOR_BGR2GRAY))

	"""
	Trims the outer white border of an image. Used for trimming pips
	"""
	def _trim(self, image):
		trimmed = np.copy(image)
		for ax in [0,1]:
			vals = np.min(np.min(trimmed, axis=2), axis=ax)
			valsAvg = np.mean(vals)
			idx1 = max(0,np.argmax(vals<valsAvg))
			if ax==0:
				idx2 = -np.argmax(vals[::-1]<valsAvg)
				if abs(idx1-idx2)>5:
					trimmed = trimmed[:,idx1:idx2]
			else:
				idx2 = -np.argmax(vals[::-1]<valsAvg)
				if idx2==0:
					idx2 = -1
				if abs(idx1-idx2)>5:
					trimmed = trimmed[idx1:idx2,:]
		return trimmed


	def _getCard(self, filename):
		im = cv2.imread(filename)
		return im

	"""
	Thresholds an image
	"""
	def _threshold(self, image):
		return cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

	"""
	Compares a score for the similarity of two images.
	Used for comparing an input image against part of the training set
	"""
	def _compare(self, im1, im2):
		im3 = cv2.absdiff(im1,im2)
		score = np.mean(im3)/np.mean(im2)
		if False:
			cv2.imshow("1",im1)
			cv2.imshow("2",im2)
			cv2.imshow("3",im3)
			cv2.moveWindow("1", 0,0)
			cv2.moveWindow("2", 0,150)
			cv2.moveWindow("3", 0,300)
			print score
			cv2.waitKey(0)
		return score

	"""
	Runs the input BlackjackCard against the 13 values and 4 suits in the
	training set. Returns a list of the most likely matches. 
	"""
	def getClosestCards(self, card, numCards=1):
		suitVals = {s:float("inf") for s in "DCHS"}
		valueVals = {v:float("inf") for v in "A23456789TJQK"}
		for i in range(2):
			for s in "DCHS":
				sScore = self._compare(card.suits[i], self.pipSuits[s])
				suitVals[s] = min(suitVals[s], sScore)
			for v in "A23456789TJQK":
				vScore = self._compare(card.values[i], self.pipValues[v])
				valueVals[v] = min(valueVals[v], vScore)
		vals = {v+s:suitVals[s]+valueVals[v] for s in "DCHS" for v in"A23456789TJQK"}
		
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
				c = cv2.blur(c, (5,5))
				card = BlackjackCard(c)
				cv2.imshow("", card.getCard())
				cv2.waitKey(0)

				#match, val = self.getClosestCard(c)
				matches, values = self.getClosestCards(card, 15)
				if matches[0] != v+s:
					if v+s in matches:
						print v+s, matches.index(v+s), matches
					else:
						print v+s, matches
		print "END", time()-t


#b = BlackjackComparer()
#b._test()