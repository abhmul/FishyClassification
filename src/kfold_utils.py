from __future__  import print_function

import shutil
import os

import numpy as np

from keras.callbacks import ModelCheckpoint


class KFoldFromDir(object):

    def __init__(self, nfolds, class_labels,
                 root='../input',
                 total_data='train',
                 train_data='train_split',
                 val_data='val_split'):

        self.nfolds = nfolds
        self.root = root
        self.train_data = train_data
        self.val_data = val_data
        self.total_data_dir = os.path.join(self.root, total_data)
        self.train_data_dir = os.path.join(self.root, self.train_data)
        self.val_data_dir = os.path.join(self.root, self.val_data)
        self.class_labels = class_labels

    @staticmethod
    def copy(img, class_label, total_dir, split_dir):
        source = os.path.join(total_dir, class_label, img)
        target = os.path.join(split_dir, class_label, img)
        try:
            shutil.copy(source, target)
            return True
        except IOError, e:
            print("Unable to copy file. %s" % e)
            return False

    @staticmethod
    def copy_folder(images, class_label, total_dir, split_dir, counter=0):

        for img in images:
            success = KFoldFromDir.copy(img, class_label, total_dir, split_dir)
            if success:
                counter += 1
        return counter

    @staticmethod
    def make_split_dir(class_label, image_split, total_dir, split_dir, split_type='', counter=0):

        split_type += ' '
        print('Rebuilding {}directory tree...'.format(split_type))
        if class_label not in os.listdir(split_dir):
            os.mkdir(os.path.join(split_dir, class_label))

        print('Copying {}split to folder {}...'.format(split_type, class_label))
        counter = KFoldFromDir.copy_folder(image_split, class_label, total_dir, split_dir, counter=counter)

        return counter

    def remove_past_splits(self):

        # Remove the train and val dirs if they already exist and create empty versions
        data_dirs = set(os.listdir(self.root))
        if self.train_data in data_dirs:
            shutil.rmtree(self.train_data_dir)
        if self.val_data in data_dirs:
            shutil.rmtree(self.val_data_dir)
        os.mkdir(self.train_data_dir)
        os.mkdir(self.val_data_dir)

    def fit(self, model_func, train_datagen, val_datagen, model_name, nbr_epochs=25,
            img_width=299, img_height=299, batch_size=32, callbacks=[]):

        for i in range(self.nfolds):
            # Clean the current split
            print('Removing past splits...')
            self.remove_past_splits()

            #  Initialize the the enumerators for train and val data
            nbr_train_samples = 0
            nbr_validation_samples = 0

            for class_label in self.class_labels:

                # List out all of the images of species fish
                print('Performing split {} for species {}'.format(i, class_label))
                total_images = np.array(os.listdir(os.path.join(self.total_data_dir, class_label)))
                inds = np.arange(len(total_images))
                # Shuffle them if this is our first fold
                np.random.shuffle(inds) if not i else None
                # get the validation start and end ind
                split_ind_begin = int(1. / self.nfolds * i * len(inds))
                split_ind_end = int(1. / self.nfolds * (i + 1) * len(inds))

                # Make the train images by making an array without the fold indices
                print('Making train split...')
                train_images = total_images[np.concatenate((inds[:split_ind_begin], inds[split_ind_end:]))]
                nbr_train_samples += self.make_split_dir(class_label,
                                                         train_images,
                                                         self.total_data_dir,
                                                         self.train_data_dir,
                                                         split_type='train')

                # Make the validation images by making an array with the fold indicies
                print('Making validation split...')
                val_images = total_images[inds[split_ind_begin:split_ind_end]]
                nbr_validation_samples += self.make_split_dir(class_label,
                                                              val_images,
                                                              self.total_data_dir,
                                                              self.val_data_dir,
                                                              split_type='validation')

            train_generator = train_datagen.flow_from_directory(
                self.train_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                shuffle=True,
                classes=self.class_labels,
                class_mode='categorical')

            validation_generator = val_datagen.flow_from_directory(
                self.val_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                shuffle=True,
                classes=self.class_labels,
                class_mode='categorical')

            # autosave best Model
            best_model_file = model_name + '_weights_fold{}.h5'
            best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                                         save_weights_only=True)

            model = model_func()
            print('Training Model...')
            model.fit_generator(
                train_generator,
                samples_per_epoch=nbr_train_samples,
                nb_epoch=nbr_epochs,
                validation_data=validation_generator,
                nb_val_samples=nbr_validation_samples,
                verbose=1,
                callbacks=[best_model] + callbacks)



