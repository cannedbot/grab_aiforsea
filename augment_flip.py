"""
Augment dataset by flipping.
prepared by haq [haqqnol@gmail.com]
"""

import os
import imutils

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
print(file_list)

for i in range(len(file_list)):
    cfile = file_list[i]
    image = cv2.imread(source_folder_to_filter + '/' + file_list[i] + '.jpg')

    copy_image = image.copy()

    width_hat = copy_image.shape[0]
    height_hat = copy_image.shape[1]

    mask_image = np.zeros((width_hat, height_hat, 3), np.uint8)



    copy_mask = mask_image.copy()
    mask_image_flipped = cv2.flip(copy_mask, 1)

    flipped_img = copy_image.copy()
    flipped_img = cv2.flip(flipped_img, 1)

    xl, yl = get_mask_coordinates(mask_image_flipped, channel='G', val=100)
    mask_flipped_output = np.zeros((int(flipped_img.shape[0]), int(flipped_img.shape[1]), 3), np.uint8)


    cv2.imwrite(dest_folder_with_accepted_dataset + "/" + cfile + "flipped" + ".jpg",
                    flipped_img)

