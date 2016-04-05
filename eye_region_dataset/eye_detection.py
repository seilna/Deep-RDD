import cv2
import numpy as np
import sys, getopt

options = getopt.getopt(sys.argv[1:], shortopts=None, longopts=["glasses"])
op_list = options[0]

snapshot_dir = './snapshot/'
snapshot_index = 0

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/data/haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/data/haarcascades/haarcascade_eye.xml')


assert len(op_list) >= 1, 'please check the usage again. You should add the option [--glasses]'
for op, args in op_list:
	print op, args
	if op == '--glasses' :
		if args == 'True':
			eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
	else :
		print 'please check the Usage. options are [--glasses ]'
		
cap = cv2.VideoCapture(0)
idx = 0

while True:
	idx += 1
	ret, frame = cap.read()
	
	frame = cv2.resize(frame, (500, 500))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#gray = cv2.resize(gray, (255,255))

	face = face_cascade.detectMultiScale(gray, 1.1, 3)
	if len(face) == 0 : 
		print 'in %dth snapshot, face not detected...!!' % idx
		continue
	else:
		print '%dth snapshot spark!!! ' % idx

	for x,y,w,h in face:
		roi_gray = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) != 2: continue
        for ex,ey,ew,eh in eyes:
			cv2.rectangle(roi_gray, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
cap.release()
cv2.destroyAllWindows()
