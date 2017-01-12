from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
from functools import partial
import logging

from predict_utils import predict_augment, predict_kfold, predict_normal

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 1000
nbr_augmentation = 5
nfolds = 5

root_path = '../input'
test_data_dir = os.path.join(root_path, 'test')

# test data generator for prediction
test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# Make the predictor
predict = partial(predict_normal, gen=test_datagen,
                  nbr_test_samples=nbr_test_samples, img_width=img_width,
                  img_height=img_height, data_dir=test_data_dir, batch_size=batch_size)
predict = partial(predict_augment, predictor=predict, nbr_aug=nbr_augmentation)
predict = partial(predict_kfold, predictor=predict)

logging.info('Running {} folds'.format(nfolds))
InceptionV3_models = []
for i in xrange(nfolds):
    logging.info('Loading model and weights from training process fold {}/{} ...'.format(i+1, nfolds))
    weights_path = os.path.join(root_path, 'inception_weights_fold{}.h5'.format(i))
    InceptionV3_models.append(load_model(weights_path))

predictions, test_image_list = predict(InceptionV3_models)

logging.info('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('image,ALB,BET,DOL,LAG,OTHER,SHARK,YFT,NoF\n') #TODO Change this later for newer models
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        logging.info('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

logging.info('Submission file successfully generated!')
