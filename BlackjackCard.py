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
	"""
	def getEdgeWhiteness(self):
		return 1

	def getCard(self):
		return self.card

	def getPips(self):
		return self.pips
