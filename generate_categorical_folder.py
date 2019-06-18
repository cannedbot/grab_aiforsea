import tarfile
import os
from scipy import io
from scipy.io import loadmat
import shutil

# choose "test" or "train" dataset to process
process_type = "train"

# extract car_devkit.tgz file from https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
labels_tar_fname = "./car_devkit.tgz"
dist_extracted_labels_dir = "./car_devkit"
car_train_img_dir = "./cars_train"
car_test_img_dir = "./cars_test"

if not os.path.isdir(dist_extracted_labels_dir):
	label_tar_file = tarfile.open(labels_tar_fname)
	label_tar_file.extractall(path=dist_extracted_labels_dir)
	print("[INFO] Done extracting!")
else:
	print("[INFO] tar folder already exist!")

mat_train = loadmat('./car_devkit/devkit/cars_train_annos.mat')
mat_test = loadmat('./car_devkit/devkit/cars_test_annos.mat')
meta = loadmat('./car_devkit/devkit/cars_meta.mat')

# print(mat_train.get("annotations")[0][0])
print(mat_train['annotations'][0])

## pre-process train dataset before generating to tfRecord file
## 1 - first generate sub folders
labels = list()

if process_type == "test":
	dst_sorted_dataet_dir = "/categorical_test_folder_car"
elif process_type == "train":
	dst_sorted_dataet_dir = "/categorical_train_folder_car"
else:
	print("process unknown!")

for l in meta['class_names'][0]:
	# issue: original "Ram C/V Cargo Van Minivan 2012" label cause cascading folder so I re-label to "Ram C-V Cargo Van
	# Minivan 2012"
	if l[0] == "Ram C/V Cargo Van Minivan 2012":
		labels.append("Ram C-V Cargo Van Minivan 2012")
		clabels = "Ram C-V Cargo Van Minivan 2012"
	else:
		labels.append(l[0])
		clabels = l[0]

	current_directory = os.getcwd()
	# print("c-", current_directory)
	target_folder = current_directory + dst_sorted_dataet_dir + "/" + clabels
	final_directory = os.path.join(current_directory, target_folder)
	# print(final_directory)

	if not os.path.exists(final_directory):
		os.makedirs(final_directory)
		print("[INFO] FOLDER CREATED")
	else:
		print("[INFO] FOLDER ALREADY EXISTED")

# print(len(labels))

train = list()

## 2 - move images to respective labeled folder
if process_type == "test":
	for example in mat_test['annotations'][0]:
		print(example)
		# print("example:", example[-2][0][0] - 1)
		# print(len(labels))
		# label = labels[example[-2][0][0] - 1]
		# print(label)
		# label = labels[example[-2][0][0] - 1]
		# image = example[-1][0]
		# shutil.copy(car_test_img_dir + '/' + image, '.' + dst_sorted_dataet_dir + "/" + label)
elif process_type == "train":
	for example in mat_train['annotations'][0]:
		# print(example)
		# print("example:", example[-2][0][0] - 1)
		# print(len(labels))
		label = labels[example[-2][0][0] - 1]
		# print(label)
		image = example[-1][0]
		# print("example:", label)
		# # train.append((image, label))
		shutil.copy(car_train_img_dir + '/' + image, '.' + dst_sorted_dataet_dir + "/" + label)
else:
	print("process unknown!")

# print(train)
