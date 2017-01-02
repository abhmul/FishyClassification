from keras.models import load_model
import os
from image2 import ImageDataGenerator
from Transformations import Rescale
from models import inception_model
from functools import partial
import logging

from predict_utils import predict_augment, predict_kfold, predict_fcn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# img_width = 299
# img_height = 299
batch_size = 32
nbr_test_samples = 1000
nfolds = 3

PosFishNames = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
NegFishNames = ['NoF']

root_path = '../input'
test_data_dir = os.path.join(root_path, 'test2')

# test data generator for prediction
# test_datagen = ImageDataGenerator(
#         rescale=1./255
        # shear_range=0.1,
        # zoom_range=0.1,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # horizontal_flip=True
# )

test_datagen = ImageDataGenerator()
test_datagen.add(Rescale(1./255))

testgen = test_datagen.flow_from_directory(test_data_dir, target_size=(None, None),
                                     batch_size=1, shuffle=False)
x = next(testgen)


model = inception_model(test=True)
for i in xrange(nfolds):
    best_model_file = '../fishyFCNInception_weights_fold{}.h5'.format(i + 1)
    model.load_weights(best_model_file)
    prediction = model.predict(x, 1)

    for i, lbl in enumerate(PosFishNames + NegFishNames):
        print 'Prediction for {}:'.format(lbl)
        print prediction[0, i]



# Make the predictor
# predict = partial(predict_normal, gen=test_datagen,
#                   nbr_test_samples=nbr_test_samples, img_width=img_width,
#                   img_height=img_height, data_dir=test_data_dir, batch_size=batch_size)