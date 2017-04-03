import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from scipy.ndimage import zoom

from models import vgg16_bb as model_func

INPUT_SHAPE = (224, 224, 3)
TARGET_SHAPE = (7, 7, 1)
DEBUG=True

# Some debugging stuff
if DEBUG:
    import matplotlib.pyplot as plt
    plt.ion()

print("Loading the data")
# Xtr = np.load("../input/train/train_imgs.npy")
# masktr = np.load("../input/train/train_masks.npy")
Xtr = np.load("../input/train/train_imgs.npy").astype(np.uint8)
masktr = np.load("../input/train/train_masks.npy").astype(np.uint8)


print("Splitting the data")
# Xtr, Xval, masktr, maskval = train_test_split(Xtr, masktr, train_size = 0.8)

# Initialize the data augmenter
print("Creating the data augmenters")
train_gen_args = dict(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

val_gen_args = dict(rescale=1./255)

train_datagen = ImageDataGenerator(**train_gen_args)
mask_datagen = ImageDataGenerator(**train_gen_args)

val_datagen = ImageDataGenerator(**val_gen_args)
mask_val_datagen = ImageDataGenerator(*val_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
img_gen = train_datagen.flow(Xtr, seed=seed)
mask_gen = mask_datagen.flow(masktr, seed=seed)
# val_seed = 2
# val_gen = val_datagen.flow(Xval, seed=val_seed)
# mask_val_gen = mask_val_datagen.flow(maskval, seed=val_seed)

# combine generators into one which yields image and masks
train_generator = zip(img_gen, mask_gen)
# val_generator = zip(val_gen, mask_val_gen)

def img_mask_generator(datagen):
    for img, mask in datagen:
        mask = zoom(mask,
        (1, TARGET_SHAPE[0] /  mask.shape[1], TARGET_SHAPE[1] /  mask.shape[2], 1),
        order=0)
        if DEBUG:
            ax[0, 0].imshow(img[0])
            ax[0, 1].imshow(mask[0, :, :, 0], cmap="gray")
            ax[1, 0].imshow(img[0] * zoom(mask[0], (INPUT_SHAPE[0] / mask.shape[1], INPUT_SHAPE[1] / mask.shape[2], 1)))
            plt.pause(1)
        yield img, mask
if DEBUG:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    for a in img_mask_generator(train_generator):
        pass

# This will save the best scoring model weights to the parent directory
best_model_file = '/output/vgg16_bb_weights.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                             save_weights_only=True)

train_steps = int((Xtr.shape[0] + 31) / 32)
val_steps = int((Xval.shape[0] + 31) / 32)

model = model_func(INPUT_SHAPE, TARGET_SHAPE)
print(model.summary())

# Train the model
print("Training the model")
model.fit_generator(img_mask_generator(train_generator),
                    steps_per_epoch=train_steps,
                    nb_epoch=50,
                    validation_data=img_mask_generator(val_generator),
                    validation_steps=val_steps,
                    verbose=1,
                    callbacks=[best_model])
