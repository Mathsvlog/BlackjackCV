import cv2
#from math import cos,sin,pi,acos
import numpy as np
#import PointFunctions as pt
from BlackjackGlobals import *

class BlackjackCard:

	def __init__(self, image):
		self.card = image
		self.pips = [image[:pipY,:pipX], image[:-pipY-1:-1,:-pipX-1:-1]]

	"""
	Place card and pip images onto the display in the correct grid location
	"""
	def displayCard(self, display, i, j):
		x,y = cardX*i, (cardY+pipY)*j
		display[y:y+cardY,x:x+cardX] = self.card
		y += cardY
		for pip in self.pips:
			display[y:y+pipY,x:x+pipX] = pip
			pipBlur = cv2.blur(pip, pipSize)
			pip = cv2.addWeighted(pip, 1+pipSharpen, pipBlur, -pipSharpen, 0)
			display[y:y+pipY,x+pipX:x+pipX*2] = pip
			x += pipX*2

	"""
	Card candidate metric: Amount of whiteness near getEdgeWhiteness
	NOT FINISHED
	"""
	def getEdgeWhiteness(self):
		x = [0, pipX, cardX-pipX, cardX]
		y = [0, pipX, cardY-pipX, cardY]
		numPixels = (cardX*cardY - (cardX-pipX*2)*(cardY-pipX*2))*255
		rgb = (0,0,0)
		idx = ((0,1),(0,2),(2,3),(1,3))
		
		for i in range(4):
			xIdx, yIdx = idx[i-1], idx[i]
			x1,x2,y1,y2 = x[xIdx[0]], x[xIdx[1]], y[yIdx[0]], y[yIdx[1]]
			part = self.card[y1:y2,x1:x2]
			currArea = np.shape(part)[0]*np.shape(part)[1]
			avg = np.average(part, axis=(0,1))*float(currArea)/numPixels
			rgb = [sum(a) for a in zip(rgb, avg)]
		
		"""
		for i in range(4):
			xIdx, yIdx = idx[i-1], idx[i]
			x1,x2,y1,y2 = x[xIdx[0]], x[xIdx[1]], y[yIdx[0]], y[yIdx[1]]
			self.card = cv2.rectangle(self.card, (x1,y1),(x2,y2), (255,0,0))
		"""

		#print "\t",rgb
		low,high = min(rgb),max(rgb)

		return high-low

	def getCard(self):
		return self.card

	def getPips(self):
		return self.pips
