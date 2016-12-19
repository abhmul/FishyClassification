import numpy as np


def predict_augment(model_list, gen, nbr_augmentation, nbr_test_samples, nbr_classes, img_width, img_height, test_data_dir,
                    batch_size=32, predictor=None):

    model = model_list[0]
    predictions = np.zeros((nbr_test_samples, nbr_classes))
    image_list = []
    for idx in range(nbr_augmentation):
        print('{}th augmentation for testing ...'.format(idx))
        if predictor is None:
            random_seed = np.random.random_integers(0, 100000)
            test_generator = gen.flow_from_directory(
                    test_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    shuffle=False, # Important !!!
                    seed=random_seed,
                    classes=None,
                    class_mode=None)

            image_list = test_generator.filenames if not idx else image_list
            print('Begin to predict for testing data ...')
            predictions += model.predict_generator(test_generator, nbr_test_samples)
        else:
            sub_predictions, image_list = predictor(model)
            predictions += sub_predictions

    return predictions / nbr_augmentation, image_list


def predict_kfold(model_list, gen, nbr_test_samples, nbr_classes, img_width, img_height, test_data_dir,
                    batch_size=32, predictor=None):

    predictions = np.zeros((nbr_test_samples, nbr_classes))
    image_list = []
    for i in range(len(model_list)):
        print('{}th fold for testing ...'.format(i))
        if predictor is None:
            random_seed = np.random.random_integers(0, 100000)
            test_generator = gen.flow_from_directory(
                test_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                shuffle=False,  # Important !!!
                seed=random_seed,
                classes=None,
                class_mode=None)

            image_list = test_generator.filenames if not i else image_list
            print('Begin to predict for testing data ...')
            predictions += model_list[i].predict_generator(test_generator, nbr_test_samples)
        else:
            sub_predictions, image_list = predictor(model_list[i])
            predictions += sub_predictions

    return predictions / len(model_list), image_list