"""
Augment dataset by rotating.
prepared by haq [haqqnol@gmail.com]
"""
import cv2
import numpy as np
import math
import os
import shutil
from HelperAugmentationFunctions import *

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

for i in range(len(file_list)):
    cfile = file_list[i]
    image = cv2.imread(source_folder_to_filter + '/' + file_list[i] + '.jpg')

    copy_image = image.copy()

    width_hat = copy_image.shape[0]
    height_hat = copy_image.shape[1]

    mask_image = np.zeros((width_hat, height_hat, 3), np.uint8)


    i_row_roi_list = []
    f_row_roi_list = []

    i_column_roi_list = []
    f_column_roi_list = []

        # (6,10,2) = 6,8
    for angle in np.arange(6, 10, 2):
        try:
            mask_image_copy = mask_image.copy()

            rotated_color = imutils.rotate(image, angle)
            rotated_color_copy = rotated_color.copy()

            rotated_mask = np.zeros((rotated_color_copy.shape[0], rotated_color_copy.shape[1], 3), np.uint8)
            rotated_mask = imutils.rotate(mask_image_copy, angle)
            rotated_mask_copy = rotated_mask.copy()

            i_row = int(rotated_color_copy.shape[0] * 0.1)
            f_row = int(rotated_color_copy.shape[0]) - int(rotated_color_copy.shape[0] * 0.1)

            i_column = int(rotated_color_copy.shape[1] * 0.1)
            f_column = int(rotated_color_copy.shape[1]) - int(rotated_color_copy.shape[1] * 0.1)

            rotated_color_roi = rotated_color_copy[i_row:f_row, i_column:f_column]
            rotated_mask_roi = rotated_mask_copy[i_row:f_row, i_column:f_column]


            #
            roi_mask_draw = np.zeros((int(rotated_mask_roi.shape[0]), int(rotated_mask_roi.shape[1]), 3),
                                         np.uint8)
            # roi_mask_draw = draw_from_coefficient(rotated_color_roi, coefficients_left, rotated_color_roi.shape[1],
                #                                       color=(255, 0, 0))


            cv2.imwrite(dest_folder_with_accepted_dataset + "/" + cfile + "rotatecw-" + str(angle) + ".jpg",
                            rotated_color_roi)

        except:
            print("fail")

            # cv2.waitKey(0)


#################################
        # (356, 360, 2) = 356, 358
    for angle in np.arange(356, 360, 2):
        try:
            mask_image_copy = mask_image.copy()

            rotated_color = imutils.rotate(image, angle)
            rotated_color_copy = rotated_color.copy()

            rotated_mask = np.zeros((rotated_color_copy.shape[0], rotated_color_copy.shape[1], 3), np.uint8)
            rotated_mask = imutils.rotate(mask_image_copy, angle)
            rotated_mask_copy = rotated_mask.copy()

            i_row = int(rotated_color_copy.shape[0] * 0.1)
            f_row = int(rotated_color_copy.shape[0]) - int(rotated_color_copy.shape[0] * 0.1)

            i_column = int(rotated_color_copy.shape[1] * 0.1)
            f_column = int(rotated_color_copy.shape[1]) - int(rotated_color_copy.shape[1] * 0.1)

            rotated_color_roi = rotated_color_copy[i_row:f_row, i_column:f_column]
            rotated_mask_roi = rotated_mask_copy[i_row:f_row, i_column:f_column]



            cv2.imwrite(dest_folder_with_accepted_dataset + "/" + cfile + "rotatecw-" + str(angle) + ".jpg",
                            rotated_color_roi)

        except:
            print("fail")