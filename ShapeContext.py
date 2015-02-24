import cv2
import numpy as np
from BlackjackGlobals import *
import PointFunctions as pf
from math import pi
from BlackjackCard import BlackjackCard

numBinsR = 8
numBinsT = 12
#maxRadius = 60
maxPoints = 100
def shapeContext(contour):
	x = map(lambda c:c[0][0], contour)
	y = map(lambda c:c[0][1], contour)
	#center = (sum(contour)/len(contour))[0]
	center = ((min(x)+max(x))/2, (min(y)+max(y))/2)
	print center
	maxRadius = max(map(lambda c:pf.dist(c[0],center),contour))
	if len(contour)>maxPoints:
		contour = contour[np.random.choice(len(contour),maxPoints)]
	bins = np.zeros((numBinsR, numBinsT))
	for i in xrange(len(contour)):
		p = contour[i][0]
		radius,theta = pf.dist(center,p), pf.theta(center,p)
		theta = theta if theta>0 else theta+2*pi
		r = int(radius*(numBinsR-1)/maxRadius)
		t = int(theta*(numBinsT-1)/2/pi)
		if radius > maxRadius:
			print "BAD RADIUS", radius
		r = r if r<numBinsR else numBinsR-1
		bins[r,t] += 1
	bins /= bins.sum()
	return bins

def shapeContextAll(contour):
	center = (sum(contour)/len(contour))[0]
	maxRadius = max(map(lambda c:pf.dist(c[0],center),contour))*2
	if len(contour)>maxPoints:
		contour = contour[np.random.choice(len(contour),maxPoints)]
	bins = np.zeros((numBinsR, numBinsT))
	for i in xrange(len(contour)):
		p1 = contour[i][0]
		for j in xrange(len(contour)):
			if i!=j:
				p2 = contour[j][0]
				radius,theta = pf.dist(p1,p2), pf.theta(p1,p2)
				theta = theta if theta>0 else theta+2*pi
				r = int(radius*(numBinsR-1)/maxRadius)
				t = int(theta*(numBinsT-1)/2/pi)
				if radius > maxRadius:
					print "BAD RADIUS", radius
				r = r if r<numBinsR else numBinsR-1
				bins[r,t] += 1
	bins /= bins.sum()
	return bins

def shapeContextDiff(sc1, sc2):
	cost = 0
	for r in xrange(numBinsR):
		for t in xrange(numBinsT):
			val1, val2 = sc1[r,t], sc2[r,t]
			cost += abs(val1-val2)
	return cost/numBinsR/numBinsT*100


def p(c):
	print map(lambda x:list(x[0]),c)

def getContours(contours, pip=None):
	value = []
	suit = []
	for c in contours:
		center,_ = cv2.minEnclosingCircle(c)
		if cv2.contourArea(c)==0 or abs(center[0]-pipX/2)/pipX>.4:
			color = (255,255,0)
		elif ((pipY-center[1])/pipY < 0.4):
			color = (255,0,0)
			suit.append(c)
		else:
			color = (0,200,0)
			value.append(c)
		if not pip is None:
			cv2.drawContours(pip, [c],0,color,1)
	return np.concatenate(value), np.concatenate(suit)

def shapeContextFromCard(filename):
	pip = cv2.resize(cv2.imread(filename), cardSize)[:pipY,:pipX]
	contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	#value, suit = getContours(contours)
	value, suit = getContours(contours,pip)
	cv2.imshow("", pip)
	return shapeContext(value), shapeContext(suit)

def test1():
	sc = cv2.createShapeContextDistanceExtractor()
	getPip = lambda name:cv2.resize(cv2.imread(name), cardSize)[:pipY,:pipX]
	getContour = lambda image:cv2.findContours(cv2.Canny(image, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

	# original images
	card1 = "2S"
	card2 = "QH"
	pip1 = getPip("train/sharp/"+card1+".jpg")
	pip2 = getPip("train/original/"+card1+".jpg")
	pip3 = getPip("train/sharp/"+card2+".jpg")
	pip4 = getPip("train/original/"+card2+".jpg")
	# contours
	c1 = getContour(pip1)
	c2 = getContour(pip2)
	c3 = getContour(pip3)
	c4 = getContour(pip4)
	# divide contours into suit/value
	v1,s1 = getContours(c1, pip1)
	v2,s2 = getContours(c2, pip2)
	v3,s3 = getContours(c3, pip3)
	v4,s4 = getContours(c4, pip4)
	# fail at shape context
	print sc.computeDistance(c2[0], c2[1])
	print sc.computeDistance(v1, v1)
	print sc.computeDistance(v1, v2)
	print sc.computeDistance(v1, v3)
	print sc.computeDistance(v1, v4)
	# show images
	cv2.imshow("1",pip1)
	cv2.imshow("2",pip2)
	cv2.imshow("3",pip3)
	cv2.imshow("4",pip4)
	cv2.waitKey(0)

	sc1 = shapeContext(v1)
	sc2 = shapeContext(v2)
	sc3 = shapeContext(v3)
	sc4 = shapeContext(v4)
	print shapeContextDiff(sc1,sc2), 1, 2
	print shapeContextDiff(sc1,sc3), 1, 3
	print shapeContextDiff(sc1,sc4), 1, 4
	print shapeContextDiff(sc2,sc3), 2, 3
	print shapeContextDiff(sc2,sc4), 2, 4
	print shapeContextDiff(sc3,sc4), 3, 4

def test2():
	shapeContextSuit = {}
	shapeContextValue = {}
	folder = "train/original/"
	#folder = "cards/"
	pipResize = (pipSize[0],int(pipSize[0]*5./3))
	#pipResize = (90,150)
	for s in "DCHS":
		f = "train/pip/"+s+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		shapeContextSuit[s] = shapeContext(np.concatenate(contours))
	"""	
		for c in contours:
			cv2.drawContours(pip, [c],0,(0,255,0),1)
		cv2.imshow(s,pip)
		#cv2.imshow(s,cv2.Canny(pip, 100, 200))
	cv2.waitKey(0)
	return
	"""


	for v in "A23456789TJQK":
		f = "train/pip/"+v+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		shapeContextValue[v] = shapeContext(np.concatenate(contours))
	"""
		for c in contours:
			cv2.drawContours(pip, [c],0,(0,255,0),1)
		cv2.imshow(v,pip)
	cv2.waitKey(0)
	return
	"""

	for s in "DCHS":
		for v in "A23456789TJQK":
			name = v+s
			value, suit = shapeContextFromCard(folder+name+".jpg")
			
			minSuitDist = float("inf")
			minSuitValue = float("inf")
			minSuit = None
			minValue = None

			suitDict = {}
			for key in shapeContextSuit.keys():
				suitDist = shapeContextDiff(suit,shapeContextSuit[key])
				suitDict[key] = suitDist
				if suitDist<minSuitDist:
					minSuitDist,minSuit = suitDist,key

			for key in shapeContextValue.keys():
				valueDist = shapeContextDiff(value,shapeContextValue[key])
				if valueDist<minSuitValue:
					minSuitValue,minValue = valueDist,key

			if name!=minValue+minSuit:
				print name, "_" if name[0]==minValue else minValue, "_" if name[1]==minSuit else (minSuit,suitDict)
			cv2.waitKey(0)
	#cv2.waitKey(0)				
	
test2()
