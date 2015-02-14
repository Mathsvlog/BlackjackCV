# card and pip dimensions
_cardScale = .5
cardSize = (int(250*_cardScale),int(350*_cardScale))
_pipAmount = (0.15, 0.25)
pipSize = tuple(map(lambda i:int(round(cardSize[i]*_pipAmount[i])),(0,1)))
cardX,cardY = cardSize# card image dimensions
pipX,pipY = pipSize# pip subimage dimensions
imageX,imageY = 640, 480# standard image size
# pip sharpen amount
pipSharpen = 1.5
