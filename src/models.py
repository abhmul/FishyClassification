from keras.applications.inception_v3 import InceptionV3
from keras.layers import AveragePooling2D, Convolution2D, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K


def inception_model(input_shape=None, fcn=True, test=False, learning_rate=0.0001, dim_ordering='default'):

    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()

    if input_shape is None:
        input_shape = (256, 256, 3) if dim_ordering == 'tf' else (3, 256, 256)

    print('Loading InceptionV3 Weights ...')
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                                    input_tensor=None, input_shape=input_shape)
    # Note that the preprocessing of InceptionV3 is:
    # (x / 255 - 0.5) x 2
    # Get the output shape
    interm_shape = InceptionV3_notop.output_shape
    global_pool = interm_shape[1:3] if dim_ordering == 'tf' else interm_shape[2:]

    # Change the input shape if we are testing
    if fcn and test:
        input_shape = (None, None, input_shape[-1]) if dim_ordering == 'tf' else (input_shape[0], None, None)
        print('Loading InceptionV3 Weights for testing ...')
        InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                                        input_tensor=None, input_shape=input_shape)

    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = InceptionV3_notop.get_layer(index=-1).output  # Shape: (*, *, 2048)
    output = AveragePooling2D(global_pool, strides=global_pool, name='avg_pool')(output)  # Shape: (1, 1, 2048)
    if fcn:
        # activation = 'sigmoid' if test else 'softmax'
        output = Convolution2D(8, 1, 1, activation='sigmoid')(output)
        if not test:
            output = Flatten(name='flatten')(output)
    else:
        output = Flatten(name='flatten')(output)
        output = Dense(8, activation='softmax', name='predictions')(output)

    InceptionV3_model = Model(InceptionV3_notop.input, output)
    # InceptionV3_model.summary()

    print('Creating optimizer and compiling')
    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return InceptionV3_model

