from keras.models import load_model
import os
from image2 import ImageDataGenerator
from Transformations import Rescale
from models import inception_model
import numpy as np
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


model = inception_model(test=True)
for k in xrange(3):
    x = next(testgen)
    for i in xrange(nfolds):
        best_model_file = '../fishyFCNInception_weights_fold{}.h5'.format(i + 1)
        model.load_weights(best_model_file)
        prediction = model.predict(x[0], 1)

        probs = np.zeros((len(PosFishNames) + len(NegFishNames)))
        for j, lbl in enumerate(PosFishNames + NegFishNames):
            print 'Prediction for {}:'.format(lbl)
            print prediction[0, :, :, j]
            a = np.max(prediction[0, :, :, j])
            b = np.min(prediction[0, :, :, j])
            print 'Max Prob {}'.format(a)
            print 'Min Prob {}'.format(b)
            if j < 7:
                probs[j] = a
            else:
                probs[j] = b
        probs = probs / np.sum(probs)
        print 'Probabilities:'
        print probs
        raw_input('Continue?')



# Make the predictor
# predict = partial(predict_normal, gen=test_datagen,
#                   nbr_test_samples=nbr_test_samples, img_width=img_width,
#                   img_height=img_height, data_dir=test_data_dir, batch_size=batch_size)