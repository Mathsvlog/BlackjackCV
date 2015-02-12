_cardScale = 0.2
_cardBox = (int(250*_cardScale),int(350*_cardScale))
_pipAmount = (0.15, 0.25)
_pipBox = tuple(map(lambda i:int(round(_cardBox[i]*_pipAmount[i])),(0,1)))
bx,by = _cardBox
px,py = _pipBox