import cv2
import numpy as np
import sys
def relative_region(glass_region_list=[]):

	def draw_point(event, x, y, flags, param):
		if event == 1:
			print "draw!"
			cv2.circle(frames, (x,y), 3, (0,255,0), 1)
			glass_region_list.append((x,y))

			
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", draw_point)
	

	face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
	cap = cv2.VideoCapture(0)
	while True:
		ret, frames = cap.read()
		frames = cv2.resize(frames, (250,250))
		face = face_cascade.detectMultiScale(frames, 1.1, 3)
		for a,b,w,h in face:
			cv2.rectangle(frames, (a,b), (a+w,b+h), (255,0,0), 1)
		cv2.imshow("image", frames)
		if ord('q') == (cv2.waitKey(20) & 0xFF): break
		

	while True:
		if len(glass_region_list) == 4:
			relative_region = []
			for a,b,w,h in face:
				for i in xrange(2):
					x1,y1 = glass_region_list[i*2]
					x2,y2 = glass_region_list[i*2+1]

					relative_width = float(x2-x1)/float(w)
					relative_height = float(y2-y1)/float(h)

					relative_start_x = float(x1-a)/float(w)
					relative_start_y = float(y1-b)/float(h)
					relative_region.append((relative_width, relative_height, relative_start_x, relative_start_y))
			if len(relative_region) != 2:
				print "relative region error"
				print relative_region
			assert(len(relative_region) == 2), "relative region error"
			return relative_region

		gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
		face = face_cascade.detectMultiScale(gray, 1.1, 3)

		"""
		for a,b,w,h in face:
			cv2.rectangle(frames, (a,b), (a+w,b+h), (255,0,0), 1)
		"""

		cv2.imshow("image", frames)
		k = cv2.waitKey(20) & 0xFF
		if k == 27: break
		elif k == ord('a'):
			print "error!!!!!!"
			sys.exit(0)
			print glass_region_list
			assert len(glass_region_list) == 4, "region fixing error"
			return glass_region_list

def glasses_region(face, relative_region_list):
	print "in getting glasses region, >> ",
	print face
	print relative_region_list
	glasses_region_list = []
	for a,b,w,h in face:
		for r_w,r_h,r_sx,r_sy in relative_region_list:
			new_sx = w*r_sx
			new_sy = h*r_sy
			new_w = r_w * w
			new_h = r_h * h
			glasses_region_list.append((new_sx, new_sy, new_w, new_h))

	print "final glasses region >> " ,
	print glasses_region_list
	assert(len(glasses_region_list) == 2), "glasses region proposal error."
	return glasses_region_list

		
		

#grl = glasses_region()
#print grl
