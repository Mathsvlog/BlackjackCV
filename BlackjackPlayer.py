import cv2
from math import cos,sin,pi,acos
import numpy as np
import sys

class BlackjackPlayer:

	def __init__(self, doRun=True):
		# init camera if exists
		self.webcam = cv2.VideoCapture(0)
		self.hasWebcam, frame = self.webcam.read()

		_bigBoxScale = 0.2
		self.bigBox = (int(250*_bigBoxScale),int(350*_bigBoxScale))

		# vector functions
		self.dist = lambda a,b:(a[0]-b[0])**2+(a[1]-b[1])**2
		self.dot = lambda a,b:a[0]*b[0]+a[1]*b[1]
		self.vec = lambda a,b:[b[0]-a[0],b[1]-a[1]]
		self.norm = lambda a,b:[a[1]-b[1],b[0]-a[0]]
		self.angle = lambda a,b,z:acos(dot(vec(a,b),vec(b,z))/(dist(a,b)*dist(b,z)))
		self.ccw = lambda a,b,z:dot(norm(a,b),vec(b,z))#>0

		if doRun:
			self.run()

	"""
	Show image in window. Pressing escape will close program.
	"""
	def showImage(self, image, name="cards", width=1000):
		# scale image
		w,h = len(image[0]),len(image)
		scale = float(width)/w
		image = cv2.resize(image, (int(w*scale),int(h*scale)))
		cv2.imshow(name, image)
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
	2.5x3.5 rectangle
	"""
	def computePerspective(box):
		# ensure box points are in circular order
		if dist(box[0],box[1])>dist(box[2],box[1]):
			for t1,t2 in [(0,1),(2,3),(1,3)]:
				for i in range(2):
					box[t1][i],box[t2][i] = box[t2][i],box[t1][i]
		# order final points depending if points are ccw 
		X = [self.bigBox[0],0,0,self.bigBox[0]]
		if (ccw(box[0],box[1],box[2])>0):
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
		self.showImage(image)

	def run(self):
		# without webcam, run computations on specific images
		if not self.hasWebcam:
			im1 = cv2.imread("cards-640.jpg")
			#analyzeImageForCards1(im1)
			analyzeImageForCards2(im1)
			for i in range(1,20):
				im = cv2.imread("frames/"+str(i)+".jpg")
				#analyzeImageForCards1(im)
				analyzeImageForCards2(im)

		# with webcam, run computations on webcam images
		if self.hasWebcam:
			blurPixels=10
			blurPixels=(blurPixels,blurPixels)
			amount = 0.5
			while True:
				_, frame =self.webcam.read()
				blur = cv2.blur(frame, blurPixels)
				#frame2 = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)
				frame2 = cv2.addWeighted(frame, 1+amount, blur, -amount, 0)
				#analyzeImageForCards2(frame2)
				self.analyzeImageForCards(frame2)

BlackjackPlayer()