import numpy as np
import os

from model import create_model, vgg_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import log_loss
from process_data import read_and_normalize_train_data, run_cross_validation_process_test
from sklearn.cross_validation import KFold
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

from keras import __version__ as keras_version


def run_cross_validation_create_models(nfolds=10, model_func=vgg_model):
    # input image dimensions
    batch_size = 64
    nb_epoch = 5
    random_state = 51

    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        model = model_func()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        imgen = ImageDataGenerator(
            # rescale=1./255,
            rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            # vertical_flip=True,
            fill_mode='nearest')
        imgen_train = imgen.flow(X_train, Y_train, batch_size=batch_size)

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        filepath = "weights-improvement-fold%s-{epoch:02d}-{val_loss:.4f}.hdf5" % num_fold
        # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0) # , checkpoint
        ]
        model.fit_generator(imgen_train, samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
                            validation_data=(X_valid, Y_valid), callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)

    return info_string, models

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
