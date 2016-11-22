
from keras.preprocessing.image import ImageDataGenerator


from model import model_dict
from GLOBALS import PATH_TRAIN_CROPPED, PATH_VAL_CROPPED, INPUT_IMGSIZE, MODE, MODEL, PLOT

import keras

print keras.__version__

# Get which model to use
create_model = model_dict[MODEL]

# Do image augmentation
tr_imgen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

# Do validation normalization
val_imgen = ImageDataGenerator(
            rescale=1./255,
            fill_mode='nearest')

# Create the model
model = create_model()

# Fit the model
history = model.fit_generator(tr_imgen.flow_from_directory(PATH_TRAIN_CROPPED, target_size=INPUT_IMGSIZE, color_mode=MODE), samples_per_epoch=1224,
                    nb_epoch=100, validation_data=val_imgen.flow_from_directory(PATH_VAL_CROPPED, target_size=INPUT_IMGSIZE, color_mode=MODE),
                    nb_val_samples=20*8)

# Plot the results
if PLOT:
    from plotter import plot_history
    plot_history(history)
