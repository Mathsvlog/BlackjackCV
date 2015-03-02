import cv2
import numpy as np
from BlackjackGlobals import *

class BlackjackCard:

	"""
	Class for an image of a playing card.
	Automatically extracts suit and value from pip corners
	"""
	def __init__(self, image):
		self.card = image
		self.pips = [np.copy(image[:pipY,:pipX]), np.copy(image[:-pipY-1:-1,:-pipX-1:-1])]
		self._computePipSharpened()
		self._extractPipParts()
		self.name = "?"

	"""
	Extract the suit and value subimages from the pip images
	"""
	def _extractPipParts(self):
		doShow = False
		self.suits = []
		self.values = []

		for i,p in enumerate(self.pipSharpened):
			if doShow:
				cv2.imshow("p1"+str(i), p)
			# trip dark pixels in corner
			numCornerTrims = 0
			while np.max(p[0,0])!=255 and numCornerTrims<3:
				p = p[1:,1:]
				numCornerTrims += 1
			# trim outer edges
			for ax in [0,1]:
				vals = np.mean(np.min(p, axis=2), axis=ax)
				valsAvg = np.mean(vals)
				valsAvg += (np.max(vals)-valsAvg)/4
				idx1 = max(0,np.argmax(vals<valsAvg))
				if ax==0:
					idx2 = idx1+np.argmax(vals[idx1:]>valsAvg)
					if abs(idx1-idx2)>5:
						p = p[:,idx1:idx2]
				else:
					idx2 = -np.argmax(vals[::-1]<valsAvg)
					if idx2==0:
						idx2 = -1
					if abs(idx1-idx2)>5:
						p = p[idx1:idx2,:]
			# separate suit and value
			vals = np.min(np.min(p, axis=2), axis=1)
			valsAvg = np.mean(vals)
			valsAvg += (np.max(vals)-valsAvg)/4
			idx1 = max(0,np.argmax(vals>valsAvg))
			idx2 = -max(0,np.argmax(vals[-1::-1]>valsAvg))-1
			size = float(np.shape(vals)[0])
			if idx1/size < .4 or idx1/size > .6:
				idx1 = int(size*.5)
			if -idx2/size < .35 or -idx2/size > .45:
				idx2 = -int(size*.4)
			value = cv2.cvtColor(cv2.resize(p[:idx1], pipPartSize), cv2.COLOR_BGR2GRAY)
			suit = cv2.resize(p[idx2:], pipPartSize)
			
			th = np.mean(value)
			value = cv2.threshold(value, th, 255, cv2.THRESH_BINARY)[1]
			th = np.mean(suit)
			suit = cv2.threshold(suit, th, 255, cv2.THRESH_BINARY)[1]
			self.values.append(value)
			self.suits.append(suit)

			if doShow:
				cv2.imshow("s"+str(i), suit)
				cv2.imshow("v"+str(i), value)
				cv2.imshow("p"+str(i), p)
		if doShow:
			cv2.waitKey(0)
	
	"""
	Sharpen the pip corners images and save as new images
	"""
	def _computePipSharpened(self):
		self.pipSharpened = []
		for pip in self.pips:
			pipBlur = cv2.blur(pip, pipSize)
			pipSharp = cv2.addWeighted(pip, 1+pipSharpen, pipBlur, -pipSharpen, 0)
			pipCanny = cv2.Canny(pipSharp, 100, 400)
			_, contours, _ = cv2.findContours(pipCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			pipContour = np.copy(pipSharp)
			self.pipSharpened.append(np.copy(pipSharp))

	"""
	Place card and pip images onto the display in the correct grid location
	"""
	def displayCard(self, display, i, j):
		x,y = cardX*i, (cardY+pipY)*j
		display[y:y+cardY,x:x+cardX] = self.card
		y += cardY
		for i in range(2):
			pip, suit, value = self.pips[i], self.suits[i], self.values[i]
			display[y:y+pipY,x:x+pipX] = pip
			dy = pipY/2
			display[y:y+pipY-dy-1,x+pipX:x+pipX*2] = cv2.cvtColor(cv2.resize(value, (pipX, dy)), cv2.COLOR_GRAY2BGR)
			display[y+dy+1:y+pipY,x+pipX:x+pipX*2] = cv2.resize(suit, (pipX, dy))
			x += pipX*2
		cv2.putText(display, self.name, (x,y+pipY/2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,255), thickness=fontThick)
	
	"""
	Name used as the identity of a playing card (as in, suit and value)
	"""
	def setCardName(self, name):
		self.name = name

	def getCard(self):
		return self.card

	def getPips(self):
		return self.pips

