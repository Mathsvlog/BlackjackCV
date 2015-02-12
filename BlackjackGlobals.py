# card and pip dimensions
_cardScale = 0.2
cardSize = (int(250*_cardScale),int(350*_cardScale))
_pipAmount = (0.15, 0.25)
pipSize = tuple(map(lambda i:int(round(cardSize[i]*_pipAmount[i])),(0,1)))
cardX,cardY = cardSize# card image dimensions
pipX,pipY = pipSize# pip subimage dimensions

# pip sharpen amount
pipSharpen = 10