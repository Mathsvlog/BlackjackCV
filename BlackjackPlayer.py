import cv2,os,sys
from math import cos,sin,pi,acos
import numpy as np
from BlackjackImage import *
import PointFunctions as pt

class BlackjackPlayer:

	def __init__(self, doRun=True, ignoreWebcam=False):
		# init camera if exists
		if ignoreWebcam:
			self.hasWebcam = False
		else:
			self.webcam = cv2.VideoCapture(0)
			self.hasWebcam, frame = self.webcam.read()

		_bigBoxScale = 0.2
		self.bigBox = (int(250*_bigBoxScale),int(350*_bigBoxScale))
		pipAmount = (0.17, 0.25)
		self.pipBox = tuple(map(lambda i:self.bigBox[i]*pipAmount[i],(0,1)))

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
		if not skipWaitKey:
			# with webcam, wait one frame
			if self.hasWebcam:
				if cv2.waitKey(1) & 0xFF == 27:
					self.webcam.release()
					cv2.destroyAllWindows()
					sys.exit()
			# without webcam, show image until keypress
			elif cv2.waitKey(0) & 0xFF == 27:
				sys.exit()

	"""
	rect = ((centerX,centerY),(width,height),(angle))
	return list of box points given rect
	"""
	def computeBoxPoints(self, rect):
		# extract rectangle params
		x,y=rect[0]
		w=rect[1][0]/2
		h=rect[1][1]/2
		a = rect[2]*pi/180.
		c,s = cos(a),sin(a)
		# create list of unrotated points
		p = (-w,w,-h,h)
		off = [(p[0],p[2]),(p[1],p[2]),(p[1],p[3]),(p[0],p[3])]
		# rotate points
		pts = map(lambda p:(
			int(round(x+p[0]*c-p[1]*s)), 
			int(round(y+p[0]*s+p[1]*c))), off)
		return np.array([pts])


	"""
	Compute the perspective matrix M to transform a box into a
	2.5x3.5 ignoreWebcam=Truee
	"""
	def computePerspective(self, box):
		if isinstance(box[0],tuple):
			box = map(lambda b:list(b),box)
		# ensure box points are in circular order
		if pt.dist(box[0],box[1])>pt.dist(box[2],box[1]):
			for t1,t2 in [(0,1),(2,3),(1,3)]:
				for i in range(2):
					box[t1][i],box[t2][i] = box[t2][i],box[t1][i]
		# order final points depending if points are ccw 
		X = [self.bigBox[0],0,0,self.bigBox[0]]
		if (pt.ccw(box[0],box[1],box[2])):
			X = [0,self.bigBox[0],self.bigBox[0],0]
		Y = [0,0,self.bigBox[1],self.bigBox[1]]
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
		return np.reshape(np.append(P,1),(3,3))



	def analyzeImageForCards(self, image):
		#self.showImage(image)
		cards = image.extractCardCandidates()
		self.displayCards(image, cards)

	"""
	Extract cards and display them in the output window
	"""
	def displayCards(self, image, cards):
		n = len(cards)
		c = 6# columns in output
		bx,by = self.bigBox
		px,py = self.pipBox
		cardDisplay = np.zeros(((by+py)*((n+c-1)/c),bx*c,3) if n>0 else (by,bx*c,3), np.uint8)
		for idx, card in enumerate(cards):
			M = self.computePerspective(card)
			imageCard = cv2.warpPerspective(image.getInputImage(), M, self.bigBox)
			x,y = bx*(idx%c), (by+py)*(idx/c)
			cardDisplay[y:y+by,x:x+bx] = imageCard[:,:]
			y = int(y+by)
			cardDisplay[y:y+py,x:x+px] = imageCard[:py,:px]
		self.showImage(cardDisplay, "BlackjackCV - Out", 75*c, True)


	def run(self):
		blurPixels=10
		blurPixels=(blurPixels,blurPixels)
		amount = 0
		# without webcam, run computations on specific images
		if not self.hasWebcam:
			for filename in map(lambda i:"images/"+str(i)+".jpg", ["cards-640"]+range(1,16)):
				im = cv2.imread(filename)
				blur = cv2.blur(im, blurPixels)
				frame2 = cv2.addWeighted(im, 1+amount, blur, -amount, 0)
				image = BlackjackImage(frame2)
				self.analyzeImageForCards(image)
				self.showImage(image)

		# with webcam, run computations on webcam images
		if self.hasWebcam:
			while True:
				_, frame =self.webcam.read()
				blur = cv2.blur(frame, blurPixels)
				#frame2 = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)
				frame2 = cv2.addWeighted(frame, 1+amount, blur, -amount, 0)
				#analyzeImageForCards2(frame2)
				image = BlackjackImage(frame2)
				self.analyzeImageForCards(image)
				self.showImage(image)

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

b = BlackjackPlayer(ignoreWebcam=True)