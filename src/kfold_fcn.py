import os
import shutil

import numpy as np

class KFoldFromDirFCN(object):
    def __init__(self, pos_class_labels,
                 neg_class_labels,
                 root='../input',
                 total_data='train',
                 pos_data='POS',
                 neg_data='NEG',
                 train_data='train_split',
                 val_data='val_split'):

        self.root = root
        self.train_data = train_data
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.val_data = val_data
        self.total_data_dir = os.path.join(self.root, total_data)
        self.train_data_dir = os.path.join(self.root, self.train_data)
        self.val_data_dir = os.path.join(self.root, self.val_data)
        self.pos_class_labels = pos_class_labels
        self.neg_class_labels = neg_class_labels

    @staticmethod
    def copy(source, target):
        try:
            shutil.copy(source, target)
            return True
        except IOError, e:
            print("Unable to copy file. %s" % e)
            return False

    @staticmethod
    def copy_folder(images, source, target, counter=0):

        for img in images:
            success = KFoldFromDirFCN.copy(os.path.join(source, img), os.path.join(target, img))
            if success:
                counter += 1
        return counter

    @staticmethod
    def make_split_dir(image_split, class_label, source, target, split_type='', counter=0):

        split_type += ' '
        print('Rebuilding {}directory tree...'.format(split_type))
        if class_label not in os.listdir(target):
            os.mkdir(os.path.join(target, class_label))

        print('Copying {}split to folder {}...'.format(split_type, class_label))
        counter = KFoldFromDirFCN.copy_folder(image_split, os.path.join(source, class_label),
                                              os.path.join(target, class_label), counter)

        return counter

    def remove_past_splits(self):

        # Remove the train and val dirs if they already exist and create empty versions
        data_dirs = set(os.listdir(self.root))
        if self.train_data in data_dirs:
            shutil.rmtree(os.path.join(self.train_data_dir, self.pos_data))
            shutil.rmtree(os.path.join(self.train_data_dir, self.neg_data))
        if self.val_data in data_dirs:
            shutil.rmtree(os.path.join(self.val_data_dir, self.pos_data))
            shutil.rmtree(os.path.join(self.val_data_dir, self.neg_data))
        os.mkdir(os.path.join(self.train_data_dir, self.pos_data))
        os.mkdir(os.path.join(self.train_data_dir, self.neg_data))
        os.mkdir(os.path.join(self.val_data_dir, self.pos_data))
        os.mkdir(os.path.join(self.val_data_dir, self.neg_data))

    def test_train_split(self, class_label, total_images, inds, split_ind_begin, split_ind_end, is_pos):

        subdir = self.pos_data if is_pos else self.neg_data

        nbr_train = 0
        nbr_val = 0
        # Make the train images by making an array without the fold indices
        print('Making train split...')
        train_images = total_images[np.concatenate((inds[:split_ind_begin], inds[split_ind_end:]))]
        nbr_train += self.make_split_dir(train_images, class_label,
                                         os.path.join(self.total_data_dir, subdir),
                                         os.path.join(self.train_data_dir, subdir),
                                         split_type='train')

        # Make the validation images by making an array with the fold indices
        print('Making validation split...')
        val_images = total_images[inds[split_ind_begin:split_ind_end]]
        nbr_val += self.make_split_dir(val_images, class_label,
                                       os.path.join(self.total_data_dir, subdir),
                                       os.path.join(self.val_data_dir, subdir),
                                       split_type='validation')

        return nbr_train, nbr_val

    def build_split(self, i, nfolds, ind_class_dict=None):

        # Clean the current split
        print('Removing past splits...')
        self.remove_past_splits()

        #  Initialize the the enumerators for train and val data
        nbr_pos_train = 0
        nbr_pos_val = 0
        nbr_neg_train = 0
        nbr_neg_val = 0

        ind_class_dict = {} if not i else ind_class_dict

        for is_pos, class_labels in [(True, self.pos_class_labels), (False, self.neg_class_labels)]:

            #  Initialize the the enumerators for train and val data
            nbr_train = 0
            nbr_val = 0
            subdir = self.pos_data if is_pos else self.neg_data

            for label in class_labels:

                # List out all of the images of species fish
                print('Performing split {} for species {}'.format(i, label))
                total_images = np.array(os.listdir(os.path.join(self.total_data_dir, subdir, label)))
                if not i:
                    ind_class_dict[label] = np.arange(len(total_images))
                    # Shuffle them if this is our first fold
                    np.random.shuffle(ind_class_dict[label])
                inds = ind_class_dict[label]
                # get the validation start and end ind
                split_ind_begin = int(1. / nfolds * i * len(inds))
                split_ind_end = int(1. / nfolds * (i + 1) * len(inds))

                t, v = self.test_train_split(label, total_images, inds, split_ind_begin, split_ind_end, is_pos)
                nbr_train += t
                nbr_val += v

            if is_pos:
                nbr_pos_train += nbr_train
                nbr_pos_val += nbr_val
            else:
                nbr_neg_train += nbr_train
                nbr_neg_val += nbr_val

        return nbr_pos_train, nbr_pos_val, nbr_neg_train, nbr_neg_val, ind_class_dict

    def fit(self, nfolds):

        ind_class_dict = {}
        for i in range(nfolds):
            nbr_pos_train, nbr_pos_val, nbr_neg_train, nbr_neg_val, ind_class_dict = \
                self.build_split(i, nfolds, ind_class_dict=ind_class_dict)

            yield nbr_pos_train, nbr_pos_val, nbr_neg_train, nbr_neg_val