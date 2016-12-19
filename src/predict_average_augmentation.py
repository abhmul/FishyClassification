from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
from functools import partial

from predict_utils import predict_augment, predict_kfold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 1000
nbr_augmentation = 5
nfolds = 5

root_path = '../'
test_data_dir = os.path.join(root_path, 'input/test/')

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# test data generator for prediction
test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

predict = partial(predict_augment, gen=test_datagen, nbr_augmentation=nbr_augmentation,
                  nbr_test_samples=nbr_test_samples, nbr_classes=len(FishNames), img_width=img_width,
                  img_height=img_height, test_data_dir=test_data_dir, batch_size=batch_size)

print('Running {} folds'.format(nfolds))
InceptionV3_models = []
for i in xrange(nfolds):
    print('Loading model and weights from training process fold {}/{} ...'.format(i+1, nfolds))
    weights_path = os.path.join(root_path, 'inception_weights_fold{}.h5'.format(i))
    InceptionV3_models.append(load_model(weights_path))

predictions, test_image_list = predict_kfold(InceptionV3_models, None, nbr_test_samples, len(FishNames),
                                             img_width, img_height,
                                             test_data_dir, batch_size=batch_size,predictor=predict)

print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
