from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Reshape, Dropout, Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.models import Model, Sequential
from keras.optimizers import SGD

from custom_metrics import dice_coef, dice_coef_loss, hard_iou, soft_iou

def vgg16_bb(input_shape, target_shape):

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    x = base_model.output
    print("Output of base model: ", base_model.output_shape)
    x = Flatten()(x)
    # let's add some fully-connected layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(target_shape[0] * target_shape[1] * target_shape[2], activation='sigmoid')(x)
    predictions = Reshape(target_shape)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss=dice_coef_loss,
                  metrics=[dice_coef, hard_iou, soft_iou, "binary_crossentropy"])
    return model

def resnet50_bb(input_shape, target_shape):

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # add a global spatial average pooling layer
    x = base_model.output
    print(base_model.output_shape)
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(target_shape[0] * target_shape[1] * target_shape[2], activation='sigmoid')(x)
    predictions = Reshape(target_shape)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss='binary_crossentropy',
                  metrics=[dice_coef, hard_iou, soft_iou])
    return model

def conv_bb(input_shape, target_shape):

    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    x = GlobalAveragePooling2D()(conv4)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(target_shape[0] * target_shape[1] * target_shape[2], activation='sigmoid')(x)
    predictions = Reshape(target_shape)(x)

    # this is the model we will train
    model = Model(input=inputs, outputs=predictions)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=[dice_coef, hard_iou, soft_iou])
    return model
