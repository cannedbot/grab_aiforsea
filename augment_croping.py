"""
Augment dataset by cropping.
prepared by haq [haqqnol@gmail.com]
"""
import os
import cv2
import math
import shutil
import numpy as np
from HelperAugmentationFunctions import *

TEST_MODE = False

file_list = []

# color dict
colorDict = {
	"blue": (255, 0, 0),
	"green": (0, 255, 0),
	"yellow": (0, 255, 255),
	"red": (0, 0, 255)
}

source_folder_to_filter = './try'
dest_folder_with_accepted_dataset = 'tryy'

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, dest_folder_with_accepted_dataset)

if not os.path.exists(final_directory):
	os.makedirs(final_directory)
	print("[INFO] FOLDER CREATED")
else:
	print("[INFO] FOLDER ALREADY EXISTED")

# make list of data names
for filename in os.listdir(source_folder_to_filter):
	if filename.endswith('.jpg'):
		fullname = os.path.join(filename)
		basename = fullname.split('.')[0]
		file_list.append(basename)
	else:
		continue

file_list.sort()

croping_percentage = 0.3

for i in range(len(file_list)):
	cfile = file_list[i]
	image = cv2.imread(source_folder_to_filter + '/' + file_list[i] + '.jpg')

	copy_image = image.copy()

	width_hat = copy_image.shape[0]
	height_hat = copy_image.shape[1]

	mask_image = np.zeros((width_hat, height_hat, 3), np.uint8)

	i_row_list, f_row_list, i_column_list, f_column_list = get_crop_params(image.shape[1], image.shape[0], crop_percentage=croping_percentage)
	roi_color = copy_image[i_column_list[i]:f_column_list[i], i_row_list[i]:f_row_list[i]]
	cv2.imwrite(dest_folder_with_accepted_dataset + "/" + cfile + "crop-" + str(i) + ".jpg", roi_color)
