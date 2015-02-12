import cv2
#from math import cos,sin,pi,acos
import numpy as np
#import PointFunctions as pt

class BlackjackCard:

	def __init__(self, image):
		self.card = image
		self.size = np.shape(self.card)
		pipAmount = (0.25, 0.15)
		self.pipSize = tuple(map(lambda i:int(round(pipAmount[i]*self.size[i])),(0,1)))
		py,px=self.pipSize
		self.pips = [image[:py,:px], image[:-py-1:-1,:-px-1:-1]]
		self.pipBlurAmount = 10

	"""
	Place card and pip images onto the display in the correct grid location
	"""
	def displayCard(self, display, i, j):
		x,y = self.size[1]*i, (self.size[0]+self.pipSize[0])*j
		display[y:y+self.size[0],x:x+self.size[1]] = self.card
		y += self.size[0]
		for pip in self.pips:
			display[y:y+self.pipSize[0],x:x+self.pipSize[1]] = pip
			blur = cv2.blur(pip, self.pipSize)
			pip = cv2.addWeighted(pip, 1+self.pipBlurAmount, blur, -self.pipBlurAmount, 0)
			display[y:y+self.pipSize[0],x+self.pipSize[1]:x+self.pipSize[1]*2] = pip
			x += self.pipSize[1]*2

	"""
	Card candidate metric: Amount of whiteness near getEdgeWhiteness
	"""
	def getEdgeWhiteness(self):
		return 1

	def getCard(self):
		return self.card

	def getPips(self):
		return self.pips
