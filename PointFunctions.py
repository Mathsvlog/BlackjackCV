from math import acos, atan2

# List of lambda functions that take in 2 points
# ab distance
dist = lambda a,b:((a[0]-b[0])**2+(a[1]-b[1])**2)**.5
# ab dot product
dot = lambda a,b:a[0]*b[0]+a[1]*b[1]
# ab vector
vec = lambda a,b:[b[0]-a[0],b[1]-a[1]]
# perpendicular of ab vector
norm = lambda a,b:[a[1]-b[1],b[0]-a[0]]
# angle between ab and bz
angle = lambda a,b,z:acos(dot(vec(a,b),vec(b,z))/(dist(a,b)*dist(b,z)))
# determines if angle between ab and bz is counterclockwise
ccw = lambda a,b,z:dot(norm(a,b),vec(b,z))>0
# theta polar coordinate of ab if a is origin
theta = lambda a,b:atan2(b[1]-a[1],b[0]-a[0])
# linear interpolation between ab
lerp = lambda a,b,t:[a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t]
# rounds numbers in a vector to nearest integer
rounded = lambda p:map(lambda i:int(round(i)), p)
# sum of ab
add = lambda a,b:[a[0]+b[0],a[1]+b[1]]

avg = lambda pts: map(lambda x:sum(x)/len(pts),zip(*pts))
center = lambda pts: map(lambda x:min(x)+(max(x)-min(x))/2,zip(*pts))