# BlackjackCV
Plays Blackjack using feed from a webcam. Made in Python with OpenCV.

Execute MAIN.py to run BlackjackPlayer.



If a webcam is accessible through the computer, the program will use live input from the camera. The camera should be pointed at a dark, clear surface that acts as the blackjack table.

Hit SPACE to calibrate the perspective transform. There should be a single playing card on the table in full view of the camera as close to the camera as possible and as centered as possible. Hit SPACE again to recompute the transform.

Hit P to throw out the current perspective transform.

Hit ESCAPE to exit the program.



If a webcam is not accessible, the program will instead go through a series of images, running the card detection and identification process on each image. The first image contains a single card and is used to calibrate the perspective transformation. Hit ESCAPE to exit the program. Hit any other key to go to the next image.

If a webcam is accessible, but you want to cycle through the images instead, change supply argment -ignorewebcam when running MAIN.py