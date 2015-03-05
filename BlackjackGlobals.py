# card and pip dimensions
_cardScale = 1
cardSize = (int(250*_cardScale),int(350*_cardScale))
_pipAmount = (0.2, 0.3)
pipSize = tuple(map(lambda i:int(round(cardSize[i]*_pipAmount[i])),(0,1)))
cardX,cardY = cardSize# card image dimensions
pipX,pipY = pipSize# pip subimage dimensions
imageX,imageY = 640, 480# standard image size
# pip sharpen amount
pipSharpen = 2

pipPartSize = (100,100)

fontSize = _cardScale
fontThick = max(1,int(_cardScale*2))

# maps card string value to card numerical value
cardValueMap = {v:10 for v in "TJQK"}
cardValueMap["A"] = 11
for v in range(2,10):
	cardValueMap[str(v)] = v

# basic strategy chart
def computeBasicStrategy():
	f = open("BasicStrategy.csv", "r")
	text = f.read()[:-1]
	f.close()
	lines = text.split("\n")
	dealer = lines[0].split(",")[1:]
	bs = {}
	for line in lines[1:]:
		player = line.split(",")
		for i,move in enumerate(player[1:]):
			bs[(player[0], dealer[i])] = move
	return bs
basicStrategy = computeBasicStrategy()