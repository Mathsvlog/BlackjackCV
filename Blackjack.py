import cv2
from math import cos,sin,pi,acos
import numpy as np
import sys

# init camera if exists
cap = cv2.VideoCapture(0)
ret, frame =cap.read()
if frame is None:
	cap = None

_bigBoxScale = 0.2
bigBox = (int(250*_bigBoxScale),int(350*_bigBoxScale))
dist = lambda x,y:(x[0]-y[0])**2+(x[1]-y[1])**2
dot = lambda x,y:x[0]*y[0]+x[1]*y[1]
vec = lambda x,y:[y[0]-x[0],y[1]-x[1]]
norm = lambda x,y:[x[1]-y[1],y[0]-x[0]]
angle = lambda x,y,z:acos(dot(vec(x,y),vec(y,z))/(dist(x,y)*dist(y,z)))
ccw = lambda x,y,z:dot(norm(x,y),vec(y,z))

def show(im, name="cards", width=1000):
	w,h = len(im[0]),len(im)
	scale = float(width)/w
	im = cv2.resize(im, (int(w*scale),int(h*scale)))
	cv2.imshow(name, im)
	if cap==None:
		if cv2.waitKey(0) & 0xFF == 27:
			sys.exit()
	else:
		if cv2.waitKey(1) & 0xFF == 27:
			cap.release()
			cv2.destroyAllWindows()
			sys.exit()

def boxPoints(rect):
	x,y=rect[0]
	w=rect[1][0]/2
	h=rect[1][1]/2
	a = rect[2]*pi/180.
	c,s = cos(a),sin(a)
	p = (-w,w,-h,h)
	off = [(p[0],p[2]),(p[1],p[2]),(p[1],p[3]),(p[0],p[3])]
	pts = map(lambda p:(
		int(round(x+p[0]*c-p[1]*s)), 
		int(round(y+p[0]*s+p[1]*c))), off)
	return np.array([pts])

def perspectiveMatrix(box):

	if dist(box[0],box[1])>dist(box[2],box[1]):
		for t1,t2 in [(0,1),(2,3),(1,3)]:
			for i in range(2):
				box[t1][i],box[t2][i] = box[t2][i],box[t1][i]

	X = [bigBox[0],0,0,bigBox[0]]
	if ccw(box[0],box[1],box[2])>0:
		X = [0,bigBox[0],bigBox[0],0]
	Y = [0,0,bigBox[1],bigBox[1]]
	B = []
	for i in range(4):
		B.append(X[i])
		B.append(Y[i])

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

	P = np.linalg.solve(A,B)
	return np.reshape(np.append(P,1),(3,3))

def extractAffineImage(brect, box, im):
	a = -rect[2]*pi/180.
	c,s = cos(a),sin(a)
	x,y = -box[0][0]
	M = np.float32([[c,-s,x*c-s*y],[s,c,x*s+c*y]])
	return cv2.warpAffine(im, M, (int(rect[1][0]),int(rect[1][1])))

def analyzeImageForCards1(im, eps=0.07):
	ret, th = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 127, 255, 1)
	image, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	imc = np.copy(im)

	found = 0
	# look at each contour
	for c in contours[1:]:
		epsilon = eps*cv2.arcLength(c, True)
		appr = cv2.approxPolyDP(c, epsilon, True)
		cv2.drawContours(imc, [c],0,(0,255,0),1)
		# if rectangle shaped and reasonably large in image
		if len(appr)==4 and 400<cv2.contourArea(c):
			#cv2.drawContours(imc, [c], 0, 255, 2300
			#print ", ".join(map(lambda a:str(a[0]), appr))

			# determine enclosing rectangle
			rect = cv2.minAreaRect(c)
			box = boxPoints(rect)
			
			# perspective transform to get rectangle
			M = perspectiveMatrix(box[0])
			im4 = cv2.warpPerspective(im, M, bigBox)
			cv2.drawContours(imc, [box],0,255,2)
			
			# overlay on original image
			try:
				w,h =bigBox
				imc[:h,w*found:w*(found+1),:] = im4
				w1,h1,off = int(w*.15),int(h*.25),int(w*.03)
				imc[h:h+h1,w*found:w*found+w1,:] = im4[off:h1+off,off:w1+off]
			except:
				break
			found += 1

	show(imc)

def analyzeImageForCards2(im):
	# get harris corners
	imGrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	dst = cv2.cornerHarris(imGrey, 5, 3, 0.04)
	#dst2 = cv2.dilate(dst, None)
	#ret, dst2 = cv2.threshold(dst2, 0.04*dst2.max(), 255, 0)
	ret, dst2 = cv2.threshold(dst, 0.04*dst.max(), 255, 0)
	dst2 = np.uint8(dst2)

	# get canny edges and save rectangular contours
	canny = cv2.Canny(im, 100, 400)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst2)
	#centroids = cv2.findContours(dst2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	_, contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	centroidList = map(lambda c:tuple(c),list(centroids))
	imOut = np.copy(im)
	cardsFound = 0
	contoursToDraw = []
	for c in contours:
		#cv2.drawContours(imOut, [c],0,(255,255,0),1)
		epsilon = 0.07*cv2.arcLength(c, True)
		appr = cv2.approxPolyDP(c, epsilon, True)

		if len(appr)==4 and 400<cv2.contourArea(c):
			#cv2.drawContours(imOut, [appr],0,(255,0,0),1)
			apprPoints = map(lambda a:tuple(a[0]), appr)

			# match contour rectangle with harris corners
			isCard = True
			prevCentroidList = centroidList[:]
			for idx in range(4):
				p = apprPoints[idx]
				closestDist = float("inf")
				closest = None
				for c in centroidList:
					currDist = dist(c,p)
					if currDist < closestDist:
						closestDist = currDist
						closest = c
				if closestDist > 10:
					isCard = False
					break
				else:
					appr[idx]=closest
					centroidList.remove(closest)
					
			# card detected
			if isCard:
				#print ",".join(map(lambda a:str(tuple(a[0])),appr))
				#cv2.drawContours(imOut, [appr],0,(255,0,0),1)
				contoursToDraw.append(appr)
				# overlay card on image
				M = perspectiveMatrix(map(lambda a:a[0], appr))
				imCard = cv2.warpPerspective(im, M, bigBox)
				w,h =bigBox
				try:
					imOut[-h:,w*cardsFound:w*(cardsFound+1),:] = imCard
					w1,h1,off = int(w*.15),int(h*.25),int(w*.03)
					imOut[-h-h1:-h,w*cardsFound:w*cardsFound+w1,:] = imCard[off:h1+off,off:w1+off]
					cardsFound += 1
				except:
					pass

			else:
				centroidList = prevCentroidList
				cv2.drawContours(imOut, [appr],0,(255,255,0),1)
	for c in contoursToDraw:
		cv2.drawContours(imOut, [c],0,(255,0,0),1)

	for x,y in centroids:
		x,y = int(x),int(y)
		cv2.circle(imOut, (x,y), 1, (0,255,0), -1)
	show(imOut)

if False:
	im1 = cv2.imread("cards-640.jpg")
	#analyzeImageForCards1(im1)
	analyzeImageForCards2(im1)
	for i in range(1,20):
			im = cv2.imread("frames/"+str(i)+".jpg")
			#analyzeImageForCards1(im)
			analyzeImageForCards2(im)


blurPixels=10
blurPixels=(blurPixels,blurPixels)
amount = 0.5
if cap!=None:
	while True:
		ret, frame =cap.read()
		blur = cv2.blur(frame, blurPixels)
		#frame2 = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)
		frame2 = cv2.addWeighted(frame, 1+amount, blur, -amount, 0)
		analyzeImageForCards2(frame2)