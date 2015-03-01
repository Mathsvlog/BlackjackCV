import cv2
import numpy as np
from BlackjackGlobals import *


class BlackjackCard:

	def __init__(self, image):
		self.card = image
		self.pips = [np.copy(image[:pipY,:pipX]), np.copy(image[:-pipY-1:-1,:-pipX-1:-1])]
		
		self._adjustPips(image)		

		self._computePipContours()
		self.name = "?"

	def _adjustPips(self,image):
		for i in range(2):
			p = self.pips[i]
			cvals = np.min(np.min(p, axis=2), axis=1)
			cvalsavg = np.mean(cvals)
			col = max(0,np.argmax(cvals<cvalsavg))/2
			rvals = np.min(np.min(p, axis=2), axis=0)
			rvalsavg = np.mean(rvals)
			row = max(0,np.argmax(rvals<rvalsavg))/2
			if i==0:
				#print col, row
				#print col, cvalsavg
				#print cvals
				self.pips[i] = np.copy(image[row:row+pipY,col:col+pipX])
			else:
				col += 1
				col = 1
				self.pips[i] = np.copy(image[:-pipY-1:-1,-col:-pipX-col:-1])

	def _computePipContours(self):
		pipContours = []
		pipSharpened = []
		pipThresholded = []
		for pip in self.pips:
			pipBlur = cv2.blur(pip, pipSize)
			pipSharp = cv2.addWeighted(pip, 1+pipSharpen, pipBlur, -pipSharpen, 0)
			pipCanny = cv2.Canny(pipSharp, 100, 400)
			_, contours, _ = cv2.findContours(pipCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			pipContour = np.copy(pipSharp)
			pipSharpened.append(np.copy(pipSharp))

			threshold = cv2.cvtColor(pipSharp, cv2.COLOR_BGR2GRAY)
			th = np.mean(threshold)
			threshold = cv2.cvtColor(cv2.threshold(threshold, th, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)
			pipThresholded.append(threshold)

			# classify contours as suit, value, and nonimportant
			for c in contours:
				center,_ = cv2.minEnclosingCircle(c)
				#if cv2.contourArea(c)==0 or abs(center[0]-pipX/2)/pipX>.4:
				if cv2.contourArea(c)==0:
					color = (255,255,0)
				elif ((pipY-center[1])/pipY < 0.4):
					color = (255,0,0)
				else:
					color = (0,200,0)
				cv2.drawContours(pipContour, [c],0,color,1)
			
			pipContours.append(pipContour)
		self.pipSharpened = pipSharpened
		self.pipContours = pipContours
		self.pipThresholded = pipThresholded

	"""
	Place card and pip images onto the display in the correct grid location
	"""
	def displayCard(self, display, i, j):
		x,y = cardX*i, (cardY+pipY)*j
		display[y:y+cardY,x:x+cardX] = self.card
		y += cardY
		for pip,pipContour in zip(self.pips,self.pipThresholded):
			display[y:y+pipY,x:x+pipX] = pip
			display[y:y+pipY,x+pipX:x+pipX*2] = pipContour
			"""
			pipBlur = cv2.blur(pip, pipSize)
			pip = cv2.addWeighted(pip, 1+pipSharpen, pipBlur, -pipSharpen, 0)
			#canny = cv2.Canny(pip, 100, 400)
			canny = cv2.cvtColor(cv2.Canny(pip, 100, 400), cv2.COLOR_GRAY2BGR)
			display[y:y+pipY,x+pipX:x+pipX*2] = canny
			"""
			x += pipX*2
		cv2.putText(display, self.name, (x,y+pipY/2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255,255,255), thickness=fontThick)
		
	def setCardName(self, name):
		self.name = name

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
