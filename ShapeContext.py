import cv2
import numpy as np
from BlackjackGlobals import *

def p(c):
	print map(lambda x:list(x[0]),c)
def drawContours(pip, contours):
	value = []
	suit = []
	for c in contours:
		center,_ = cv2.minEnclosingCircle(c)
		if ((pipY-center[1])/pipY < 0.4):
			color = (255,0,0)
			suit.append(c)
		else:
			color = (255,255,0)
			value.append(c)
		cv2.drawContours(pip, [c],0,color,1)
		#p(c)

	return np.concatenate(value), np.concatenate(suit)

sc = cv2.createShapeContextDistanceExtractor()
print type(sc)
getPip = lambda name:cv2.resize(cv2.imread(name), cardSize)[:pipY,:pipX]
getContour = lambda image:cv2.findContours(cv2.Canny(image, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

card1 = "2S"
card2 = "JD"
pip1 = getPip("train/sharp/"+card1+".jpg")
pip2 = getPip("train/original/"+card1+".jpg")
pip3 = getPip("train/sharp/"+card2+".jpg")
pip4 = getPip("train/original/"+card2+".jpg")

c1 = getContour(pip1)
c2 = getContour(pip2)
c3 = getContour(pip3)
c4 = getContour(pip4)
v1,s1 = drawContours(pip1, c1)
v2,s2 = drawContours(pip2, c2)
v3,s3 = drawContours(pip3, c3)
v4,s4 = drawContours(pip4, c4)

print sc.computeDistance(c2[0], c2[1])
print sc.computeDistance(v1, v1)
print sc.computeDistance(v1, v2)
print sc.computeDistance(v1, v3)
print sc.computeDistance(v1, v4)

cv2.imshow("1",pip1)
cv2.imshow("2",pip2)
cv2.imshow("3",pip3)
cv2.imshow("4",pip4)
cv2.waitKey(0)