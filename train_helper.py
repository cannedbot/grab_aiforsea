# reads the length of label txt file
import tensorflow as tf
import vgg_preprocessing
import resnet_v1
import glob

image_size = resnet_v1.resnet_v1.default_image_size

def file_len(filename):
	return sum(1 for line in open(filename))

# count number of tfrecord
def get_num_records_tfecords(filenames):
	counter = 0
	for fname in filenames:
		for file in tf.python_io.tf_record_iterator(fname):
			counter += 1
			# print("record:", record)
	return counter


def parse_function(example_proto):
	features = {
		'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
		'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
		'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
		'image/height': tf.FixedLenFeature((), tf.int64),
		'image/width': tf.FixedLenFeature((), tf.int64),
	}

	parsed_features = tf.parse_single_example(example_proto, features)

	image = tf.image.decode_jpeg(parsed_features["image/encoded"])
	width = tf.cast(parsed_features["image/width"], tf.int32)
	height = tf.cast(parsed_features["image/height"], tf.int32)
	label = tf.cast(parsed_features["image/class/label"], tf.int32)

	# Reshape image data into the original shape
	# image = tf.decode_raw(image, tf.uint8)
	image = tf.reshape(image, [height, width, 3])
	image = tf.cast(image, tf.float32)


	# Images need to have the same dimensions for feeding the network
	image = vgg_preprocessing.preprocess_image(image, image_size, image_size)

	return image, label

def get_batched_training_dataset(dataset_fname, bsize):
	# Load datasets
	print("Loading dataset")
	train_filenames = glob.glob(dataset_fname)
	train_dataset = tf.data.TFRecordDataset(train_filenames)
	train_dataset = train_dataset.map(parse_function)
	train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle

	return train_dataset.batch(bsize)

def get_batched_val_dataset(dataset_fname, bsize):
	val_filenames = glob.glob(dataset_fname)
	val_dataset = tf.data.TFRecordDataset(val_filenames)
	val_dataset = val_dataset.map(parse_function)

	return val_dataset.batch(bsize)