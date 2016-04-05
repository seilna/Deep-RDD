import cv2
import numpy as np
import subprocess
import sys, getopt

options = getopt.getopt(sys.argv[1:], shortopts=None, longopts=["drowsiness="])
op_list = options[0]
snapshot_dir = None
print 'please enter your name. >>> ',
name = raw_input()

assert len(op_list) == 1, 'please check the implementation usage.'

for op, args in op_list:
	print op, args
	if op == '--drowsiness' :
		if args == 'True':
			snapshot_dir = '/Users/naseil/Desktop/Hanyang University/Projects/Drowsiness_Detector/eye_region_dataset/snapshot_dir/' + name + '_drowsiness/'
		elif args == 'False':
			snapshot_dir = '/Users/naseil/Desktop/Hanyang University/Projects/Drowsiness_Detector/eye_region_dataset/snapshot_dir/' + name + '/'


subprocess.call(['mkdir', snapshot_dir])

print 'snapshot directory >> ' + snapshot_dir

a = raw_input()
snapshot_index = 0


cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	
	frame = cv2.resize(frame, (500,500))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#cv2.imshow('frame',gray)
	filename = snapshot_dir + str(snapshot_index) + '.jpeg'
	cv2.imwrite(filename, frame)
	snapshot_index += 1
	if snapshot_index % 100 == 0:
		print '%dth snapshot saved...' % snapshot_index
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
cap.release()
cv2.destroyAllWindows()



'''
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/data/haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12_2/data/haarcascades/haarcascade_eye.xml')

im = cv2.imread('face.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


face = face_cascade.detectMultiScale(gray, 1.3, 5)

for x,y,w,h in face:
	cv2.rectangle(im, (x,y), (x+w, y+h), (255,0,0), 2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = im[y:y+h, x:x+w]

	eyes = eye_cascade.detectMultiScale(roi_gray)
	for ex,ey,ew,eh in eyes:
		cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

cv2.imshow('img', im)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
