from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
import keras

import fish8 as f8
from model import model_dict
from window_processing import validate_model
from GLOBALS import MODEL, ROTATION, SHEAR, ZOOM, HORIZONTAL, VERTICAL, FILL_MODE, NUM_EPOCHS, PATH_TRAIN_CROPPED
from GLOBALS import INPUT_IMGSIZE, MODE, PLOT

print(keras.__version__)

# Get which model to use
create_model = model_dict[MODEL]

# Do image augmentation
tr_imgen = ImageDataGenerator(
            rescale=1./1,
            rotation_range=ROTATION,
            shear_range=SHEAR,
            zoom_range=ZOOM,
            horizontal_flip=HORIZONTAL,
            vertical_flip=VERTICAL,
            fill_mode=FILL_MODE)

# Load the validation data
Xval, yval, val_id = f8.load_train_data()

model = create_model()

epochs = NUM_EPOCHS
stats = {}

for i in xrange(epochs):

    print('Epoch %s/%s' % (i + 1, NUM_EPOCHS))

    # Fit the model
    print('Training the model...')
    history = model.fit_generator(tr_imgen.flow_from_directory(PATH_TRAIN_CROPPED, target_size=INPUT_IMGSIZE, color_mode=MODE),
                                  samples_per_epoch=1224, nb_epoch=1)

    stats['loss'] = history.history['loss']
    stats['accuracy'] = history.history['accuracy']

    # Run the cross validation
    print('Validating the Model...')
    stats['val_loss'] = validate_model(model, Xval, yval, batch_size=64)
    print('val loss: %s' % stats['val_loss'])

# Plot the results
if PLOT:
    from plotter import plot_metric_dict
    plot_metric_dict(stats)



