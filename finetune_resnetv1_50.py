"adopted from https://gist.github.com/d3rezz"

import tensorflow as tf
import numpy as np
import os
import glob
from tqdm import tqdm


import resnet_v1
from train_helper import *


from tensorflow.contrib import slim

# constants
DATASET_DIR = "./car/"
_FILE_PATTERN = 'car_%s_*.tfrecord'

batch_size = 32
num_epochs = 20
image_size = resnet_v1.resnet_v1.default_image_size
print(image_size*image_size*3*16)

# tensorboard summaries files to be saved
logdir = "logs/"
os.makedirs(logdir, exist_ok=True)

graph = tf.Graph()
with graph.as_default():
	tf.logging.set_verbosity(tf.logging.INFO)
	train_dataset_fname = DATASET_DIR + _FILE_PATTERN % ("train")
	val_dataset_fname = DATASET_DIR + _FILE_PATTERN % ("validation")

	batched_train_dataset = get_batched_training_dataset(train_dataset_fname, batch_size)

	batched_val_dataset = get_batched_val_dataset(val_dataset_fname, batch_size)

	train_filenames = glob.glob(train_dataset_fname)
	val_filenames = glob.glob(val_dataset_fname)

	num_classes = file_len(os.path.join(DATASET_DIR, "labels.txt"))
	num_train_records = get_num_records_tfecords(train_filenames)
	print("Loaded train dataset with %d images belonging to %d classes" % (num_train_records, num_classes))
	num_batches = np.ceil(num_train_records / batch_size)

	num_val_records = get_num_records_tfecords(val_filenames)
	print("Loaded val dataset with %d images belonging to %d classes" % (num_val_records, num_classes))

	# iterator
	iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
	                                           batched_train_dataset.output_shapes)
	images, labels = iterator.get_next()
	print("image_shape:", images.shape)
	print("labels:", labels.shape)

	train_init_op = iterator.make_initializer(batched_train_dataset)
	val_init_op = iterator.make_initializer(batched_val_dataset)

	with slim.arg_scope(resnet_v1.resnet_arg_scope()):
		logits, _ = resnet_v1.resnet_v1_50(images,
		                                   num_classes=num_classes,
		                                   is_training=True)
	logits = tf.squeeze(logits)

	variables_to_restore = tf.contrib.framework.get_variables_to_restore(
		exclude=["resnet_v1_50/logits", "resnet_v1_50/AuxLogits"])
	init_fn = tf.contrib.framework.assign_from_checkpoint_fn("./resnet_v1_50/resnet_v1_50.ckpt", variables_to_restore)

	logits_variables = tf.contrib.framework.get_variables("resnet_v1_50/logits") + tf.contrib.framework.get_variables(
		"resnet_v1_50/AuxLogits")
	logits_init = tf.variables_initializer(logits_variables)

	# Loss function:
	predictions = tf.to_int32(tf.argmax(logits, 1))
	tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	total_loss = tf.losses.get_total_loss()

	temp = set(tf.all_variables())
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
	logits_train_op = optimizer.minimize(total_loss,
	                                     var_list=logits_variables)  # use this op to only train the last layer
	full_train_op = optimizer.minimize(total_loss)  # use this op to train the whole network

	# this needs to come after defining the training op
	adam_init_op = tf.initialize_variables(set(tf.all_variables()) - temp)

	# Define the metric and update operations (taken from http://ronny.rest/blog/post_2017_09_11_tf_metrics/)
	tf_metric, tf_metric_update = tf.metrics.accuracy(labels, predictions, name="accuracy_metric")

	# Isolate the variables stored behind the scenes by the metric operation
	running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_metric")

	# Define initializer to initialize/reset running variables
	running_vars_initializer = tf.variables_initializer(var_list=running_vars)

	acc_summary = tf.summary.scalar('accuracy', tf_metric)

	# To save the trained model
	saver = tf.train.Saver()

	tf.get_default_graph().finalize()

with tf.Session(graph=graph) as sess:
	# Initializations
	init_fn(sess)
	sess.run(logits_init)
	sess.run(adam_init_op)

	print("Writing summaries to %s" % logdir)
	train_writer = tf.summary.FileWriter(os.path.join(logdir, "train/"), sess.graph)
	val_writer = tf.summary.FileWriter(os.path.join(logdir, "valid/"), sess.graph)

	# Training
	for epoch in range(num_epochs):
		print('Starting training epoch %d / %d' % (epoch + 1, num_epochs))
		# initialize the iterator with the training set
		sess.run(train_init_op)

		pbar = tqdm(total=num_batches)  # progress bar showing how many batches remain
		while True:
			try:
				# train on one batch of data
				_ = sess.run(full_train_op)
				pbar.update(1)

			except tf.errors.OutOfRangeError:
				# print("error")
				break
		pbar.close()

		# Compute training and validation accuracy
		sess.run(train_init_op)
		# initialize/reset the accuracy running variables
		sess.run(running_vars_initializer)

		while True:
			try:
				sess.run(tf_metric_update)
			except tf.errors.OutOfRangeError:
				break
		train_acc = sess.run(tf_metric)
		summary = sess.run(acc_summary)
		print('Train accuracy: %f' % train_acc)
		train_writer.add_summary(summary, epoch + 1)
		train_writer.flush()

		sess.run(val_init_op)

		# initialize/reset the accuracy running variables
		sess.run(running_vars_initializer)

		while True:
			try:
				sess.run(tf_metric_update)
			except tf.errors.OutOfRangeError:
				break
		# Calculate the score
		val_acc = sess.run(tf_metric)
		summary = sess.run(acc_summary)
		print('Val accuracy: %f' % val_acc)
		val_writer.add_summary(summary, epoch + 1)
		val_writer.flush()

		# Save every 10 epoch
		if epoch % 10 == 0:
			print("saving epoch", epoch, " checkpoint")
			saver.save(sess, os.path.join(logdir, "./saved/car_resnet_v1_50"+ str(epoch) +".ckpt"))

	# Save model
	saver.save(sess, os.path.join(logdir, "./saved/car_resnet_v1_50"+ str(epoch) +".ckpt"))
