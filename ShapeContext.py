import cv2
import numpy as np
from BlackjackGlobals import *
import PointFunctions as pf
from math import pi,log
from BlackjackCard import BlackjackCard
from random import seed

seed(0)
numBinsR = 4
numBinsT = 4
#maxRadius = 60
maxPoints = 50
radiusLog = 2*(2**.5)+1.01
def shapeContext(contour):
	return shapeContextAll(contour)

def shapeContextOne(contour):
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
		theta = theta if theta>=0 else theta+2*pi
		#r = int(radius*(numBinsR-1)/maxRadius)
		r = int(log(radius,maxRadius)*(numBinsR))
		t = int(theta*(numBinsT)/2/pi)
		if radius > maxRadius:
			print "BAD RADIUS", radius
		r = r if r<numBinsR else numBinsR-1
		bins[r,t] += 1
	bins /= bins.sum()
	return bins

def shapeContextAll(contour):
	x = map(lambda c:c[0][0], contour)
	y = map(lambda c:c[0][1], contour)
	center = ((min(x)+max(x))/2., (min(y)+max(y))/2.)
	rx, ry = max(x)-center[0], max(y)-center[1]
	#if len(contour)>maxPoints:
	#	contour = contour[np.random.choice(len(contour),maxPoints)]
	points = map(lambda p:pf.vec(center, tuple(p[0])),contour)
	points = map(lambda p:(p[0]/rx, p[1]/ry),points)
	pointBins = {}
	for i,p1 in enumerate(points):
		bins = np.zeros((numBinsR, numBinsT))
		for p2 in points[:i]+points[i+1:]:
			radius,theta = pf.dist(p1,p2), pf.theta(p1,p2)
			theta = theta if theta>=0 else theta+2*pi
			r = int(log(1+radius,radiusLog)*numBinsR)
			t = int(theta*(numBinsT)/2/pi)
			#print r,t,radius,theta, p1, p2
			bins[r,t] += 1
		bins /= bins.sum()
		pointBins[p1] = bins
	return pointBins

def shapeContextMatch(pt1, pt2):
	cost = 0
	
	for r in xrange(numBinsR):
		for t in xrange(numBinsT):
			val1, val2 = pt1[r,t], pt2[r,t]
			cost += (val1-val2)**2/(val1+val2+1)
	
	#cost = {key: d1[key] - d2.get(key, 0) for key in d1.keys()}
	#cost = sum(keypt1[keypt1]+1, pt2[r,t]+1)
	return cost

def shapeContextDiff(sc1, sc2):
	oneToTwo = {}
	twoToOne = {}
	dists1 = {p:3 for p in sc1.keys()}
	dists2 = {p:3 for p in sc2.keys()}
	for p1 in sc1.keys():
		for p2 in sc2.keys():
			d = pf.dist(p1,p2)
			if dists1[p1] > d:
				dists1[p1] = d
				oneToTwo[p1] = p2
			if dists2[p2] > d:
				dists2[p2] = d
				twoToOne[p2] = p1
	numPoints = len(dists1)+len(dists2)
	shapeCost = 0
	distCost = sum(dists1.values()) + sum(dists2.values())/numPoints
	for p1 in sc1.keys():
		shapeCost += shapeContextMatch(sc1[p1], sc2[oneToTwo[p1]])
	for p2 in sc2.keys():
		shapeCost += shapeContextMatch(sc2[p2], sc1[twoToOne[p2]])
	shapeCost /= numPoints

	return shapeCost

def angleDistance(contour):
	x = map(lambda c:c[0][0], contour)
	y = map(lambda c:c[0][1], contour)
	center = ((min(x)+max(x))/2., (min(y)+max(y))/2.)
	maxRadius = max(map(lambda c:pf.dist(c[0],center),contour))
	distances = [0]*(numBinsT)
	for i in xrange(len(contour)):
		p = contour[i][0]
		radius,theta = pf.dist(center,p)/maxRadius, pf.theta(center, p)
		theta = theta if theta>=0 else theta+2*pi
		t = int(theta*(numBinsT)/2/pi)
		distances[t] = max(distances[t], radius)
	return distances

def angleDistanceDiff(ad1, ad2):
	return 100*sum(map(lambda i: abs(ad1[i]-ad2[i]), xrange(numBinsT)))/numBinsT

def contourCenter(contour):
	x = map(lambda c:c[0][0], contour)
	y = map(lambda c:c[0][1], contour)
	center = ((min(x)+max(x))/2, (min(y)+max(y))/2)
	return center

def compareContours(contour1, contour2):
	center1, center2 = contourCenter(contour1), contourCenter(contour2)
	radius1 = max(map(lambda c:pf.dist(c[0],center1),contour1))
	radius2 = max(map(lambda c:pf.dist(c[0],center2),contour2))
	c1 = (contour1-center1)/radius1
	c2 = (contour2-center2)/radius2
	if len(c1)<len(c2):
		c1,c2=c2,c1
	score = 0
	for i in xrange(len(c1)):
		score += min(map(lambda c:pf.dist(c[0],c1[i,0]),c2))
	return score/len(c1)


def p(c):
	print map(lambda x:list(x[0]),c)

def getContours(contours, pip=None):
	value = []
	suit = []

	for c in contours:
		center,_ = cv2.minEnclosingCircle(c)
		#if cv2.contourArea(c)==0 or center[0]/pipX>.8:
		if center[0]/pipX>.8:
			color = (255,255,0)
		elif ((pipY-center[1])/pipY < 0.45):
			color = (0,255,0)
			suit.append(c)
		else:
			color = (255,0,0)
			value.append(c)
		if not pip is None:
			cv2.drawContours(pip, [c],0,color,1)
			#for pt in c:
			#	cv2.circle(pip, tuple(pt[0]), 1, color, -1)

	#value, suit = np.concatenate(value), np.concatenate(suit)
	value, suit = uniformContour(value), uniformContour(suit)
	if not pip is None:
		for pt in value:
			cv2.circle(pip, tuple(pt[0]), 1, (0,255,0), -1)
		for pt in suit:
			cv2.circle(pip, tuple(pt[0]), 1, (255,0,0), -1)

	return value, suit

def uniformContour(contours):
	lengthTotal = sum(map(lambda c:cv2.arcLength(c,False), contours))
	lengthPart = lengthTotal/maxPoints
	sampled = []
	distTotal = 0
	c = 0
	p = 0
	while c < len(contours):
		lastPoint = contours[c][0][0]
		while p < len(contours[c])-1:
			p += 1
			currPoint = contours[c][p][0]
			dist = pf.dist(lastPoint, currPoint)
			while distTotal+dist >= lengthPart:
				t = (lengthPart-distTotal) / dist				
				newPoint = pf.lerp(lastPoint, currPoint, t)
				sampled.append([newPoint])
				lastPoint = newPoint
				dist = pf.dist(lastPoint, currPoint)
				distTotal = 0
			distTotal += dist
			lastPoint = currPoint
			
		p = 0
		c += 1
		
	return sampled

def contoursFromCard(filename, show):
	card = BlackjackCard(cv2.resize(cv2.imread(filename), cardSize))
	pip = card.pipThresholded[0]
	#pip = cv2.resize(cv2.imread(filename), cardSize)[:pipY,:pipX]
	contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	
	if show:
		value, suit = getContours(contours,pip)
		cv2.imshow("", pip)
	else:
		value, suit = getContours(contours)
	return value,suit

def shapeContextFromCard(filename, show):
	value, suit = contoursFromCard(filename, show)
	if len(value)==0 or len(suit)==0:
		return None, None
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
	# use matchshapes instead
	m,p = 1,0.0
	print cv2.matchShapes(v1,v2,m,p)
	print cv2.matchShapes(v1,v3,m,p)
	print cv2.matchShapes(v1,v4,m,p)
	print cv2.matchShapes(v2,v3,m,p)
	print cv2.matchShapes(v2,v4,m,p)
	print cv2.matchShapes(v3,v4,m,p)
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
	contourSuit = {}
	contourValue = {}
	folder = "train/original/"
	#folder = "cards/"
	#pipResize = (int(pipSize[1]*3./5),int(pipSize[1]))
	pipResize = (90*2,150*2)
	preview = False#
	for s in "DCHS":
		f = "train/pip/"+s+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		#contoursAll = np.concatenate(contours)
		contoursAll = uniformContour(contours)
		contourSuit[s] = contoursAll
		shapeContextSuit[s] = shapeContext(contoursAll)
		
		if preview:
			for c in contours:
				cv2.drawContours(pip, [c],0,(255,0,0),1)
			for pt in contoursAll:
				cv2.circle(pip, tuple(pt[0]), 1, (0,255,0), -1)
			cv2.imshow(s,pip)
	if preview:
		cv2.waitKey(0)
		#return
	
	for v in "A23456789TJQK":
		f = "train/pip/"+v+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		#contoursAll = np.concatenate(contours)
		contoursAll = uniformContour(contours)
		contourValue[v] = contoursAll
		shapeContextValue[v] = shapeContext(contoursAll)
		if preview:
			for c in contours:
				cv2.drawContours(pip, [c],0,(255,0,0),1)
			for pt in contoursAll:
				cv2.circle(pip, tuple(pt[0]), 1, (0,255,0), -1)
			cv2.imshow(v,pip)
	if preview:
		cv2.waitKey(0)
		return
	
	matchMethod = 3
	for s in "DCHS":
		for v in "A23456789TJQK":
			name = v+s
			value, suit = shapeContextFromCard(folder+name+".jpg", True)# show
			if value==None or suit==None:
				print name, "UNCATEGORIZABLE"
				cv2.waitKey(0)
				continue
			minSuitDist = float("inf")
			minSuitValue = float("inf")
			minSuit = None
			minValue = None

			suitDict = {}
			for key in shapeContextSuit.keys():
				suitDist = shapeContextDiff(suit,shapeContextSuit[key])
				#suitDist = cv2.matchShapes(suit,contourSuit[key],matchMethod,0)
				suitDict[key] = suitDist
				if suitDist<minSuitDist:
					minSuitDist,minSuit = suitDist,key
			
			valueDict = {}
			for key in shapeContextValue.keys():
				valueDist = shapeContextDiff(value,shapeContextValue[key])
				#valueDist = cv2.matchShapes(value,contourValue[key],matchMethod,0)
				valueDict[key] = valueDist
				if valueDist<minSuitValue:
					minSuitValue,minValue = valueDist,key

			if name!=minValue+minSuit:
				#print name, "_" if name[0]==minValue else minValue, "_" if name[1]==minSuit else (minSuit,suitDict)
				print name, "_" if name[0]==minValue else (minValue,sorted(valueDict.items(), key=lambda x:x[1])), "_" if name[1]==minSuit else (minSuit,suitDict)
				#cv2.waitKey(0)
			

def test3():
	angleDistanceSuit = {}
	angleDistanceValue = {}
	folder = "train/original/"
	pipResize = (int(pipSize[1]*3./5),int(pipSize[1]))
	for s in "DCHS":
		f = "train/pip/"+s+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		contoursAll = np.concatenate(contours)
		angleDistanceSuit[s] = angleDistance(contoursAll)

	for v in "A23456789TJQK":
		f = "train/pip/"+v+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		contoursAll = np.concatenate(contours)
		angleDistanceValue[v] = angleDistance(contoursAll)

	matchMethod = 1
	for s in "DCHS":
		for v in "A23456789TJQK":
			name = v+s
			value, suit = contoursFromCard(folder+name+".jpg", False)
			adValue, adSuit = angleDistance(value), angleDistance(suit)
			minSuitDist = float("inf")
			minSuitValue = float("inf")
			minSuit = None
			minValue = None

			suitDict = {}
			for key in angleDistanceSuit.keys():
				suitDist = angleDistanceDiff(adSuit, angleDistanceSuit[key])
				suitDict[key] = suitDist
				if suitDist<minSuitDist:
					minSuitDist,minSuit = suitDist,key

			for key in angleDistanceValue.keys():
				valueDist = angleDistanceDiff(adValue, angleDistanceValue[key])
				if valueDist<minSuitValue:
					minSuitValue,minValue = valueDist,key

			if name!=minValue+minSuit:
				print name, "_" if name[0]==minValue else minValue, "_" if name[1]==minSuit else (minSuit,suitDict)
			#cv2.waitKey(0)

def test4():
	momentsSuit = {}
	momentsValue = {}
	folder = "train/original/"
	pipResize = (int(pipSize[1]*3./5),int(pipSize[1]))
	for s in "DCHS":
		f = "train/pip/"+s+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		contoursAll = np.concatenate(contours)
		momentsSuit[s] = cv2.moments(contoursAll)

	for v in "A23456789TJQK":
		f = "train/pip/"+v+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		contoursAll = np.concatenate(contours)
		momentsValue[v] = cv2.moments(contoursAll)

	nuKeys = ["nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"]
	for s in "DCHS":
		for v in "A23456789TJQK":
			name = v+s
			value, suit = contoursFromCard(folder+name+".jpg", False)
			value, suit = cv2.moments(value), cv2.moments(suit)
			minSuitDist = float("inf")
			minSuitValue = float("inf")
			minSuit = None
			minValue = None

			suitDict = {}
			for key in momentsSuit.keys():
				suitDist = sum(map(lambda nu:(suit[nu]-momentsSuit[key][nu])**2, nuKeys))
				suitDict[key] = suitDist
				if suitDist<minSuitDist:
					minSuitDist,minSuit = suitDist,key

			for key in momentsValue.keys():
				valueDist = sum(map(lambda nu:(value[nu]-momentsValue[key][nu])**2, nuKeys))
				if valueDist<minSuitValue:
					minSuitValue,minValue = valueDist,key

			if name!=minValue+minSuit:
				print name, "_" if name[0]==minValue else minValue, "_" if name[1]==minSuit else (minSuit,suitDict)
			#cv2.waitKey(0)

def test5():
	contourSuit = {}
	contourValue = {}
	folder = "train/original/"
	pipResize = (int(pipSize[1]*3./5),int(pipSize[1]))
	for s in "DCHS":
		f = "train/pip/"+s+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		contoursAll = np.concatenate(contours)
		if len(contoursAll)>maxPoints:
			contoursAll = contoursAll[np.random.choice(len(contoursAll),maxPoints)]
		contourSuit[s] = contoursAll

	for v in "A23456789TJQK":
		f = "train/pip/"+v+".jpg"
		pip = cv2.resize(cv2.imread(f), pipResize)
		contours = cv2.findContours(cv2.Canny(pip, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
		contoursAll = np.concatenate(contours)
		if len(contoursAll)>maxPoints:
			contoursAll = contoursAll[np.random.choice(len(contoursAll),maxPoints)]
		contourValue[v] = contoursAll

	nuKeys = ["nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"]
	for s in "DCHS":
		for v in "A23456789TJQK":
			name = v+s
			value, suit = contoursFromCard(folder+name+".jpg", False)
			if len(value)>maxPoints:
				value = value[np.random.choice(len(value),maxPoints)]
			if len(suit)>maxPoints:
				suit = suit[np.random.choice(len(suit),maxPoints)]			
			minSuitDist = float("inf")
			minSuitValue = float("inf")
			minSuit = None
			minValue = None

			suitDict = {}
			for key in contourSuit.keys():
				suitDist = compareContours(suit, contourSuit[key])
				suitDict[key] = suitDist
				if suitDist<minSuitDist:
					minSuitDist,minSuit = suitDist,key

			for key in contourValue.keys():
				valueDist = compareContours(value, contourValue[key])
				if valueDist<minSuitValue:
					minSuitValue,minValue = valueDist,key

			if name!=minValue+minSuit:
				print name, "_" if name[0]==minValue else minValue, "_" if name[1]==minSuit else (minSuit,suitDict)
			#cv2.waitKey(0)
	
test2()
#import cProfile
#cProfile.run("test2()")