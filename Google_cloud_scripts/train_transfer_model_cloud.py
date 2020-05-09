from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras import Model, layers
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
import pandas as pd
import random



# folders
train_dir = './../small_dataset/train/'
val_dir = './../small_dataset/val/'
test_dir = './../small_dataset/test/'

# Settings
train_local = False
train_real = True
show_confusion_matrix = True
plot_statistics = False
random.seed(10)

# Constants
if train_real:
    IMG_SIZE = 224  # double check in report
    LEARNING_RATE = 2e-5
    EPOCHS = 30
    BATCH_SIZE = 8
    FACTOR = 0.7
    PATIENCE = 5
    OPTIMIZER = 'Adam'

if train_local:
    IMG_SIZE = 224
    LEARNING_RATE = 2e-5
    EPOCHS = 1
    BATCH_SIZE = 8
    FACTOR = 0.7
    PATIENCE = 5
    OPTIMIZER = 'Adam'


def create_generators():
    datagen_augmented = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='constant',
        cval=0.,
        rescale=1./255
    )

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        fill_mode='constant',
        cval=0.,
        rescale=1./255
    )

    train_generator = datagen_augmented.flow_from_directory(
        train_dir,
        batch_size=BATCH_SIZE,
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='categorical'
    )

    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def build_model(unfrozen_layers):
    # Create VGG-model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Create our tensors
    last_tensor_vgg = vgg_model.output
    last_tensors = layers.Flatten()(last_tensor_vgg)
    last_tensors = layers.Dense(1024, activation='relu')(last_tensors)
    last_tensors = layers.Dense(1024, activation='relu')(last_tensors)
    last_tensors = layers.Dense(3, activation='softmax')(last_tensors)

    # Add VGG-tensors with the last tensors and create model object
    model = Model(input=vgg_model.input, output=last_tensors)

    # Freeze VGG layers
    model = freeze_layers(model, unfrozen_layers)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    return model


def freeze_layers(model, unfrozen_layers):
    for layer in model.layers[:-unfrozen_layers]:
        layer.trainable = False
    return model


def train_model(model, train_generator, val_generator):
    nb_train_samples = train_generator.n
    nb_val_samples = val_generator.n
    # The weights can be altered to pay more attention to recall/precision/accuracy. Balanced will yield better recall.
    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_generator.classes),
        train_generator.classes
    )

    trained_model = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=nb_val_samples // BATCH_SIZE,
        verbose=0,
        class_weight=class_weights,
        callbacks=[ReduceLROnPlateau(
            monitor='loss',
            factor=FACTOR,
            patience=PATIENCE,
            verbose=0)]
    )

    return trained_model


def plot_loss_accuracy(statistics):
    # summarize history for accuracy
    plt.plot(statistics['accuracy'])
    plt.plot(statistics['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'Val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(statistics['loss'])
    plt.plot(statistics['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Val'], loc='upper left')
    plt.show()


def plot_confusion_matrix(Y_pred, Y_true):
    conf_matrix = confusion_matrix(Y_true, Y_pred)
    df_cm = pd.DataFrame(conf_matrix, columns=np.unique(Y_true), index=np.unique(Y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(3, 3))
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
    plt.show()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df_cm)



if __name__ == "__main__":
    # Get generators
    train_generator, val_generator, test_generator = create_generators()

    # Build model
    model = build_model(unfrozen_layers=4)
    model.summary()

    # Train model and returns statistics
    model_statistics = train_model(model, train_generator, val_generator)
    print("Model trained")

    # Get predicted values and plot confusion matrix
    if show_confusion_matrix:
        Y_pred = model.predict_generator(val_generator)
        Y_pred = np.argmax(Y_pred, 1)
        Y_true = val_generator.classes
        plot_confusion_matrix(Y_pred, Y_true)

    # Plot statistics
    if plot_statistics:
        plot_loss_accuracy(model_statistics.history)

    # Print statistics (train & val)
    print(model_statistics.history)

    results_file = open('results.txt', 'a')
    results_file.write('Nr epochs = ' + str(EPOCHS)+ '\n')
    results_file.write('Batch size = ' + str(BATCH_SIZE)+ '\n')
    results_file.write(str(model_statistics.history) + '\n')
    results_file.close()


# TODO:
# - Weight the data / should all data be augmented?
# - What optimizer function?
