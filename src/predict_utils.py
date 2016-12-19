import numpy as np
import logging


def normalize_multiclass(arr, axis=1):

    sums = np.sum(arr, axis=axis)
    sums = np.tile(sums.transpose(), (1, 3))
    return arr / sums


# Standard Predictors
def predict_normal(model_list, gen, nbr_test_samples, img_width, img_height, data_dir, batch_size=32):

    random_seed = np.random.random_integers(0, 100000)
    model = model_list[0]
    test_generator = gen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,  # Important !!!
        seed=random_seed,
        classes=None,
        class_mode=None)

    image_list = test_generator.filenames
    logging.info('Begin to predict normally for testing data ...')
    predictions = model.predict_generator(test_generator, nbr_test_samples)

    return predictions, image_list


def predict_fcn(model_list, gen, nbr_test_samples, data_dir, neg_class=-1):

    random_seed = np.random.random_integers(0, 100000)
    model = model_list[0]
    nbr_classes, predictions = None, None
    test_generator = gen.flow_from_directory(
        data_dir,
        batch_size=1,
        shuffle=False,  # Important !!!
        seed=random_seed,
        classes=None,
        class_mode=None)
    image_list = test_generator.filenames
    logging.info('Begin to predict normally for testing data...')
    for i in range(nbr_test_samples):
        sample = next(test_generator)
        activations = model.predict_on_batch(sample)  # Shape is (1, ?, ?, 8)
        if not i:
            nbr_classes = activations.shape[-1]
            predictions = np.zeros((nbr_test_samples, nbr_classes))
        activations = activations.reshape((-1, nbr_classes))
        if neg_class == -1:
            prediction = np.max(activations, axis=0)
        else:
            prediction = np.concatenate((np.max(activations[:neg_class], axis=0),
                                        np.min(activations[neg_class:neg_class+1], axis=0),
                                        np.max(activations[neg_class+1:], axis=0)))

        predictions[i] = prediction

    return normalize_multiclass(predictions), image_list


# higher order predictors
def predict_augment(model_list, predictor, nbr_aug):

    predictions, image_list = None, None
    for i in range(nbr_aug):
        logging.info('{}th augmentation for testing ...'.format(i))
        sub_predictions, image_list = predictor(model_list)
        if not i:
            predictions = sub_predictions
        else:
            predictions += sub_predictions

    return predictions / nbr_aug, image_list


def predict_kfold(model_list, predictor):

    predictions, image_list = None, None
    for i in range(len(model_list)):
        logging.info('{}th fold for testing ...'.format(i))
        sub_predictions, image_list = predictor(model_list)
        if not i:
            predictions = sub_predictions
        else:
            predictions += sub_predictions

    return predictions / len(model_list), image_list
