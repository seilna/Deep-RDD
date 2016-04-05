import cv2
import glob

face_module_path = '/usr/local/Cellar/opencv/2.4.12_2/data/haarcascades/'

face_cascade = cv2.CascadeClassifier(face_module_path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(face_module_path + 'haarcascade_eye.xml')

file_path = './usaul_dataset/'
file_idx = 0
image_dir_list = glob.glob('./snapshot_dir/*')

											
for image_dir in image_dir_list:								
	if image_dir.find('drowsiness') != -1 : continue
	print 'image directory >> ' + image_dir
	image_list = glob.glob(image_dir + '/*')

	prev_faces = None
	prev_eyes = None

	length = len(image_list)



	for index in xrange(length):
		draw_flag = False
		file_name = image_dir +'/' + str(index) + '.jpeg'
		im = cv2.imread(file_name)

		face = face_cascade.detectMultiScale(im, 1.1,5)


		for x,y,w,h in face:
			prev_faces = face
			face_region = im[y:y+h, x:x+w]
			eyes = eye_cascade.detectMultiScale(face_region)
		
			if len(eyes) != 2: continue
			prev_eyes = eyes
			for ex,ey,ew,eh in eyes:
				draw_flag = True
				#cv2.rectangle(face_region, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
				#cv2.imshow('dasdas', face_region[ey:ey+eh, ex:ex+ew])
				#cv2.waitKey()

				output_file_name = file_path + str(file_idx) + '.jpeg'
				print 'output file name >> ' + output_file_name
				cv2.imwrite(output_file_name, face_region[ey:ey+eh, ex:ex+ew])
				file_idx += 1
			#output_file_name = image_dir + 'extracted_' + str(index) + '.jpeg'
			
			if draw_flag == True: 
				break

		if draw_flag == False and prev_faces != None and prev_eyes != None:
			for x,y,w,h in prev_faces:
				face_region = im[y:y+h, x:x+w]
				
				for ex,ey,ew,eh in prev_eyes:

					output_file_name = file_path + str(file_idx) + '.jpeg'
					cv2.imwrite(output_file_name, face_region[ey:ey+eh, ex:ex+ew])
					file_idx += 1

					#cv2.rectangle(face_region, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)
			
					#cv2.imshow('dasd', face_region[ey:ey+eh, ex:ex+ew])
					#cv2.waitKey()

		"""
		if prev_eyes != None and prev_faces != None:
			for ex,ey,ew,eh in prev_eyes:
				cv2.imshow('dsad', prev_faces[ey:ey+eh, ex:ex+ew])
				cv2.waitKey()
		"""
		#cv2.imshow('dsad', im)
		#		cv2.waitKey()
