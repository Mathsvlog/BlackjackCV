dist = lambda a,b:((a[0]-b[0])**2+(a[1]-b[1])**2)**.5
dot = lambda a,b:a[0]*b[0]+a[1]*b[1]
vec = lambda a,b:[b[0]-a[0],b[1]-a[1]]
norm = lambda a,b:[a[1]-b[1],b[0]-a[0]]
angle = lambda a,b,z:acos(dot(vec(a,b),vec(b,z))/(dist(a,b)*dist(b,z)))
ccw = lambda a,b,z:dot(norm(a,b),vec(b,z))>0