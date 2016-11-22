
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from model import create_model
from GLOBALS import PATH_TRAIN_CROPPED, PATH_VAL_CROPPED

tr_imgen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=180,
            featurewise_center=True,
            featurewise_std_normalization=True,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.3,
            zoom_range=0.4,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

val_imgen = ImageDataGenerator(
            rescale=1./255,
            featurewise_center=True,
            featurewise_std_normalization=True,
            fill_mode='nearest')

model = create_model()

history = model.fit_generator(tr_imgen.flow_from_directory(PATH_TRAIN_CROPPED, color_mode='greyscale'), samples_per_epoch=1260,
                    nb_epoch=100, validation_data=val_imgen.flow_from_directory(PATH_VAL_CROPPED, color_mode='greyscale'),
                    nb_val_samples=20*8)

# Plot the results
plt.plot(history.history['val_loss'],'o-')
plt.plot(history.history['loss'],'o-')
plt.xlabel('Number of Iterations')
plt.ylabel('Categorical Crossentropy')
plt.title('Train Error vs Number of Iterations')

plt.show()
