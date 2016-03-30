import glob

total = 0
dir_list = glob.glob('./snapshot_dir/*')
for dir in dir_list:
	file_list = glob.glob(dir+'/*')
	print 'len >> ',
	print len(file_list)
	print dir
	total += len(file_list)

print 'total training dataset >>',
print total
