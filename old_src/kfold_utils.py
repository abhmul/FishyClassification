from __future__  import print_function

import shutil
import os
import unittest as tst

import numpy as np


class KFoldFromDir(object):

    def __init__(self, nfolds, class_labels,
                 root='../input',
                 total_data='train',
                 train_data='train_split',
                 val_data='val_split',
                 seed=None):

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

    def fit(self, train_datagen, val_datagen,
            img_width=None, img_height=None, batch_size=32, save_dir=None, seed=None):

        np.random.seed(seed) if seed is not None else None

        ind_class_dict = {}
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
                if not i:
                    ind_class_dict[class_label] = np.arange(len(total_images))
                    # Shuffle them if this is our first fold
                    np.random.shuffle(ind_class_dict[class_label])
                inds = ind_class_dict[class_label]
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

            if img_height is None and img_width is None:
                target_size = None
            else:
                target_size = (img_height, img_width)

            train_generator = train_datagen.flow_from_directory(
                self.train_data_dir,
                target_size=target_size,
                batch_size=batch_size,
                shuffle=True,
                classes=self.class_labels,
                class_mode='categorical',
                save_to_dir=save_dir,
                seed=np.random.randint(1, 10000))

            validation_generator = val_datagen.flow_from_directory(
                self.val_data_dir,
                target_size=target_size,
                batch_size=batch_size,
                shuffle=True,
                classes=self.class_labels,
                save_to_dir=save_dir,
                class_mode='categorical',
                seed=np.random.randint(1, 10000))

            yield (train_generator, validation_generator), (nbr_train_samples, nbr_validation_samples)


class TestKFoldMethods(tst.TestCase):

    def testInit(self):
        kf = KFoldFromDir(10, ['a', 'b'], root='../input',
                          total_data='train',
                          train_data='train_split',
                          val_data='val_split')
        self.assertEqual(10, kf.nfolds)
        self.assertEqual(['a', 'b'], kf.class_labels)
        self.assertEqual('../input', kf.root)
        self.assertEqual('../input/train', kf.total_data_dir)
        self.assertEqual('../input/train_split', kf.train_data_dir)
        self.assertEqual('../input/val_split', kf.val_data_dir)

    def testRemoveCreate(self):
        nfolds = 10
        FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
        root = '../input'
        total_data = 'train'
        train_data = 'train_split'
        val_data = 'val_split'

        kf = KFoldFromDir(nfolds, FishNames, root=root,
                          total_data=total_data,
                          train_data=train_data,
                          val_data=val_data)

        def copy_stuff():

            for class_label in FishNames:
                total_images = np.array(os.listdir(os.path.join(kf.total_data_dir, class_label)))
                train_inds = np.array([i for i in range(len(total_images)) if i % 5 == 0])
                val_inds = np.array([i for i in range(len(total_images)) if i % 5 == 1])
                train_images = total_images[train_inds]
                val_images = total_images[val_inds]

                nbr_train = kf.make_split_dir(class_label, train_images, kf.total_data_dir, kf.train_data_dir,
                                              split_type='train')
                nbr_val = kf.make_split_dir(class_label, val_images, kf.total_data_dir, kf.val_data_dir,
                                            split_type='val')

                self.assertEqual(len(train_inds), nbr_train)
                self.assertEqual(len(val_inds), nbr_val)

                self.assertEqual(os.listdir(os.path.join(root, train_data, class_label)), list(train_images))
                self.assertEqual(os.listdir(os.path.join(root, val_data, class_label)), list(val_images))

        def remove_stuff():
            kf.remove_past_splits()
            data_dirs = set(os.listdir(root))
            # Check if remove splits actually recreated the dirs1
            self.assertTrue(train_data in data_dirs)
            self.assertTrue(val_data in data_dirs)

            # Make sure the dirs are empty
            self.assertTrue(len(os.listdir(kf.train_data_dir)) == 0)
            self.assertTrue(len(os.listdir(kf.val_data_dir)) == 0)

        if len(os.listdir(kf.train_data_dir)) == 0 and len(os.listdir(kf.val_data_dir)) == 0:
            copy_stuff()
            remove_stuff()
        else:
            remove_stuff()
            copy_stuff()



if __name__ == '__main__':
    tst.main()