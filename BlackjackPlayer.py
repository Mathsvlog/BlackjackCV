import cv2,os,sys
from math import cos,sin,pi,acos,log
import numpy as np
from BlackjackImage import BlackjackImage
from BlackjackCard import BlackjackCard
from BlackjackComparer import BlackjackComparer
import PointFunctions as pt
from BlackjackGlobals import *
import operator
import PointFunctions as pf
from time import sleep
from BlackjackState import BlackjackState
from BlackjackSpeaker import BlackjackSpeaker

class BlackjackPlayer:

	printCards = False
	sharpenCardOutput = False

	"""
	Main class for playing Blackjack using a webcam or with input images
	"""
	def __init__(self, doRun=True, ignoreWebcam=False):
		self.comparer = BlackjackComparer()
		# initialize camera if one exists
		if ignoreWebcam:
			self.hasWebcam = False
		else:
			self.camAtt = cv2.CAP_PROP_SHARPNESS
			self.camAttD = 10
			self.camAttI = 50

			self.webcam = cv2.VideoCapture(0)
			if self.camAttI>=0:
				self.webcam.set(self.camAtt, self.camAttI)
			self.hasWebcam, frame = self.webcam.read()
		
		self.project = not self.hasWebcam# True to apply a projection transformation
		self.reproject = False# set True to recompute the projection transformation
		self.speaker = BlackjackSpeaker()# uses speech to declare moves

		if doRun:
			self.run()

	"""
	Show image in window. Pressing escape will close program.
	"""
	def showImage(self, image, name="BlackjackCV", width=900, skipWaitKey=False):
		if (isinstance(image, BlackjackImage)):
			image.show(name=name, width=width)
		else:
			w,h = len(image[0]),len(image)
			scale = float(width)/w
			image = cv2.resize(image, (int(w*scale),int(h*scale)))
			cv2.imshow(name, image)
		# move windows to right location
		if name=="BlackjackCV":
			cv2.moveWindow(name, 0,0)
		else:
			cv2.moveWindow(name, 918,0)

		if not skipWaitKey:
			# with webcam, wait one frame
			if self.hasWebcam:
				key = cv2.waitKey(1) & 0xFF
				if key == 27:
					self.webcam.release()
					cv2.destroyAllWindows()
					sys.exit()
				elif key == ord('e'):
					value = self.webcam.get(self.camAtt) + self.camAttD
					self.webcam.set(self.camAtt, value)
					print value, self.webcam.get(self.camAtt)
				elif key == ord('d'):
					value = self.webcam.get(self.camAtt) - self.camAttD
					self.webcam.set(self.camAtt, 0)
					print value, self.webcam.get(self.camAtt)
				elif key == ord('p'):
					BlackjackImage._projectionTransform = None
					self.project = False
				elif key == ord('w'):
					self.hasWebcam = not self.hasWebcam
				elif key==32:# SPACE
					self.project = True
					self.reproject = True
			# without webcam, show image until keypress
			else:
				key = cv2.waitKey(0) & 0xFF
				if key == 27:# ESCAPE
					sys.exit()


	"""
	Compute the perspective matrix M to transform a box into a 2.5x3.5
	If image is provided, test for card orientation
	"""
	def computePerspective(self, box, image=None, badOrientationDetected=False):
		if isinstance(box[0],tuple):
			box = map(lambda b:list(b),box)
		# rotate box so that card pips are on correct corners
		if badOrientationDetected:
			box = box[1:]+[box[0]]

		# order final points depending if points are ccw 
		X = [cardX,0,0,cardX]
		if (pt.ccw(box[0],box[1],box[2])):
			X = [0,cardX,cardX,0]
		Y = [0,0,cardY,cardY]
		# construct B from AX = B
		B = []
		for i in range(4):
			B.append(X[i])
			B.append(Y[i])
		# construct A from AX = B
		x,y = zip(*box)
		A = np.zeros((8,8))
		for i in range(4):
			i2 = i*2
			for j in range(2):
				A[i2+j][0+j*3] = x[i]
				A[i2+j][1+j*3] = y[i]
				A[i2+j][2+j*3] = 1
			A[i2][6] = -X[i]*x[i]
			A[i2][7] = -X[i]*y[i]
			A[i2+1][6] = -Y[i]*x[i]
			A[i2+1][7] = -Y[i]*y[i]
		# solve AX = B and build projection matrix
		P = np.linalg.solve(A,B)
		M = np.reshape(np.append(P,1),(3,3))
		# test card orienation

		if not self.project and not image is None:
			card = cv2.warpPerspective(image.getInputImage(), M, cardSize)
			if not self._correctCardOrientation(card):
				return self.computePerspective(box, image=None, badOrientationDetected=True)
		return M

	"""
	Determines if a card is rotated correctly based on the coloring of the pip corners
	"""
	def _correctCardOrientation(self, card):
		pips = [card[:pipX,:pipX], card[-pipX:,-pipX:], card[-pipX:,:pipX], card[:pipX,-pipX:]]
		vals = map(lambda p:sum(cv2.mean(p)), pips)
		return (vals[0]+vals[1]-vals[2]-vals[3])<0

	"""
	Given an input image, extract card candidates and analyze candidates
	for identity of cards
	"""
	def analyzeImageForCards(self, image):
		candidates, cardGroups = image.extractCardCandidates()
		cards = self.getTransformedCardCandidates(image, candidates)
		for c in cards:
			# optional card sharpening, only affects output appearance
			if BlackjackPlayer.sharpenCardOutput:
				cardImage = c.getCard()
				s,b = 1,5
				blur = cv2.blur(cardImage, (b,b))
				sharp = cv2.addWeighted(cardImage, 1+s, blur, -s*.8, 0)
				c.card = sharp
			# get closest card and set the card's name as suit+value letters
			names, values = self.comparer.getClosestCards(c, 5)
			name, value = names[0], values[0]
			certainty = str(int(log(value, .15)))
			c.setCardName(name)

			if BlackjackPlayer.printCards:
				print name, value, names, c.center
		self.displayCards(cards)
		self.cards = cards
		self.currState = BlackjackState(cards, cardGroups)
		if self.hasWebcam:
			self.speaker.analyzeState(self.currState)

	"""
	Takes in a list of card candidates and transforms them to be
	perfectly rectangular cards
	"""
	def getTransformedCardCandidates(self, image, candidates):
		cards = []
		y,x,_ = np.shape(image.getInputImage())
		order = {}
		for cand in candidates:
			M = self.computePerspective(cand, image)
			card = cv2.warpPerspective(image.getInputImage(), M, cardSize)
			centerX,centerY = reduce(lambda a,b:(a[0]+b[0], a[1]+b[1]), cand)
			centerX,centerY = centerX/4., centerY/4.
			percX, percY = (x-centerX)/x, (y-centerY)/y
			cardFinal = BlackjackCard(card, (centerX, centerY))
			order[cardFinal] = -int(percY*3)-percX
			cards.append(cardFinal)
		
		cardsSorted = sorted(order.items(), key=operator.itemgetter(1))# TODO lambda instead
		cardsSorted = [c[0] for c in cardsSorted]
		return cards

	"""
	Extract cards and display them in the output window
	"""
	def displayCards(self, cards):
		n = len(cards)
		c = 6# columns in output
		r = 5# minimum number of rows in output
		# build display image
		cardDisplay = np.zeros(((cardY+pipY)*max([((n+c-1)/c),r]),cardX*c,3), np.uint8)
		# put each card on the display
		for idx, imageCard in enumerate(cards):
			imageCard.displayCard(cardDisplay, idx%c, idx/c)
		# display the cards
		self.showImage(cardDisplay, "BlackjackCV - Out", 110*c, True)

	"""
	Alternate running mode that is only used for saving images
	"""
	def runCapture(self):
		idx = 1
		while os.path.isfile("images/"+str(idx)+".jpg"):
			idx = idx+1
		def saveImageIfClick(event,x,y,flags,param):
			if event==1:
				print param
				cv2.imwrite("images/"+str(param)+".jpg", frame)
				idx = param+1
		while True:
			_, frame =self.webcam.read()
			self.showImage(frame, width=640)
			cv2.setMouseCallback("BlackjackCV", saveImageIfClick, idx)
			while os.path.isfile("images/"+str(idx)+".jpg"):
				idx = idx+1

	"""
	Runs the Blackjack player. Uses a webcam if one exists.
	Otherwise, runs algorithm on a series of input images
	"""
	def run(self):
		blurPixels=10
		blurPixels=(blurPixels,blurPixels)
		amount = 0
		# without webcam, run computations on specific images
		if not self.hasWebcam:
			for filename in map(lambda i:"images/"+str(i)+".jpg", range(1,9)):
				im = cv2.imread(filename)
				blur = cv2.blur(im, blurPixels)
				frame2 = cv2.addWeighted(im, 1+amount, blur, -amount, 0)
				image = BlackjackImage(frame2, project=self.project, recomputeProjection=False)
				print filename
				self.analyzeImageForCards(image)
				self.showImage(image)

		# with webcam, run computations on webcam images
		if self.hasWebcam:
			while True:
				_, frame =self.webcam.read()
				image = BlackjackImage(frame, project=self.project, recomputeProjection=self.reproject)
				self.reproject = False
				self.analyzeImageForCards(image)
				self.showImage(image)

