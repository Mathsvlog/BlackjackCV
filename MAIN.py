from BlackjackPlayer import BlackjackPlayer
import sys

ignore = False
if len(sys.argv)==2:
	if sys.argv[1].lower()=="-ignorewebcam":
		ignore = True
	else:
		print "Supply argument -ignorewebcam to use images instead of accessing webcam"

b = BlackjackPlayer(ignoreWebcam=ignore)