# Grab_aiforsea
Submission for AI for SEA competition for grab (computer vision challenge)

Finetune Resnet_v1_50 architecture for stanford cars dataset classification. (architecture base code on tf-slim model zoo) 

Original stanford train set were split 0.8% for training and 0.2% for validation. Fine-tuning the architecture to the dataset results with ~0.9 train accuracy and ~0.75 validation accuracy.

(note: this was run and tested natively on my alienware aurora R6 with GTX1080)

## **Data preprocessing**
* Download and extract dataset 
	* http://imagenet.stanford.edu/internal/car196/cars_train.tgz
	  extract it to folder cars_train
	
  * Download and extract labels
	  https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
	  extract it to folder car_devkit
	
  * Clean dataset (some images are grayscale)
	  run script filter_out_bad_data.py
	
  * Augment dataset
	  run augment_flip.py, augment_croping.py and augment_rotating.py
	
  * Create categorical_folder
	  run generate_categorical_folder.py
	
  * Generate tf.record
	  follow this repo to generate tf.record https://github.com/cannedbot/create_tfrecords
    
## **To start fine-tune training**
	. run finetune_resnetv1_50.py
