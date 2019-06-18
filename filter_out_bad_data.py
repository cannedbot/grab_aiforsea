"""
convert grayscale images to RGB in a directory
"""
import cv2
import os



path = "./cars_train"
bad_image_list = list()

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
	    img = cv2.imread(path + "/" + file, cv2.IMREAD_UNCHANGED)
	    dimensions = img.shape
	    # channels = img.shape[2]
	    if dimensions != len(3):
		    bad_image_list.append(file)
		    print("bad dim")
		    print(file)
		    print('Number of Channels : ', dimensions)
		    backtorgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		    status = cv2.imwrite(path + "/" + file, backtorgb)

print(len(bad_image_list))
print(bad_image_list)

with open('bad_image.txt', 'w') as f:
    for item in bad_image_list:
        f.write("%s\n" % item)