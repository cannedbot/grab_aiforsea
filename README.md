# Grab_aiforsea
Submission for AI for SEA competition for grab (computer vision challenge)

(note: this was run and tested natively on my alienware aurora R6 with GTX1080)

## **Data preprocessing**
* download and extract dataset 
	* http://imagenet.stanford.edu/internal/car196/cars_train.tgz
	  extract it to folder cars_train
	
  * download and extract labels
	  https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
	  extract it to folder car_devkit
	
  * clean dataset (some images are grayscale)
	  run script filter_out_bad_data.py
	
  * augment dataset
	  run augment_flip.py, augment_croping.py and augment_rotating.py
	
  * create categorical_folder
	  run generate_categorical_folder.py
	
  * generate tf.record
	  follow this repo to generate tf.record https://github.com/cannedbot/create_tfrecords
    
## **To start fine-tune training**
	. run finetune_resnetv1_50.py
