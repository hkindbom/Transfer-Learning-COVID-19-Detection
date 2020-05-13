from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras import Model, layers
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
import pandas as pd
import random

# Settings
small_dataset = True
large_dataset = False
train_real = False
save_model = False
test_model = True
show_confusion_matrix = True
plot_statistics = True

# Experiments
run_experiment1 = True
run_experiment2 = False
run_experiment3 = False

# datasets, folders and paths
test_dir = 'data/dataset/largeDataset/test/' # should always be test of large data set, ignore small test data 
saved_model_path = "./transfer_model_"
saved_accuracy_plot_path = "./accuracy_plot_"
saved_loss_plot_path = "./loss_plot_"
saved_confusion_matrix_path = "./confusion_matrix_"

if small_dataset:
    dataset = 'small'
    train_dir = './data/dataset/smallDataset/train/'
    val_dir = './data/dataset/smallDataset/val/'
    weights = 'balanced'

if large_dataset:
    dataset = 'large'
    train_dir = './data/dataset/largeDataset/train/'
    val_dir = './data/dataset/largeDataset/val/'
    weights = {0: 1, 1: 1, 2: 12}

# Constants
if train_real:
    IMG_SIZE = 224 
    LEARNING_RATE = 2e-5
    EPOCHS = 22
    BATCH_SIZE = 8
    FACTOR = 0.7
    PATIENCE = 5
    OPTIMIZER = 'Adam'

else:
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


def build_model(unfrozen_layers, weights):
    # Create VGG-model
    vgg_model = VGG16(weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

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

    class_weights = class_weight.compute_class_weight(
    weights,
    np.unique(train_generator.classes),
    train_generator.classes
    ) 

    trained_model = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=nb_val_samples // BATCH_SIZE,
        verbose=1,
        class_weight=class_weights,
        callbacks=[ReduceLROnPlateau(
            monitor='loss',
            factor=FACTOR,
            patience=PATIENCE,
            verbose=0)]
    )

    return trained_model


def plot_loss_accuracy(statistics, model_name):
    range_EPOCHS = np.array(range(1, EPOCHS+1))

    plt.plot(range_EPOCHS, statistics['accuracy'])
    plt.plot(range_EPOCHS, statistics['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'Val'], loc='upper left')
    plt.savefig(saved_accuracy_plot_path + model_name + ".png")
    plt.show()

    plt.plot(range_EPOCHS, statistics['loss'])
    plt.plot(range_EPOCHS, statistics['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Val'], loc='upper left')
    plt.savefig(saved_loss_plot_path + model_name + ".png")
    plt.show()


def plot_confusion_matrix(model, model_name, generator):
    Y_pred = model.predict_generator(generator)
    Y_pred = np.argmax(Y_pred, 1)
    Y_true = generator.classes

    conf_matrix = confusion_matrix(Y_true, Y_pred)
    df_cm = pd.DataFrame(conf_matrix, columns=np.unique(Y_true), index=np.unique(Y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
    plt.savefig(saved_confusion_matrix_path + model_name + ".png")
    plt.show()

def test_model(model, model_name, test_generator):
    results = model.evaluate_generator(test_generator)
    print(str(model.metrics_names[1]) + " of " + str(model_name) + " on test data: ", results[1])
    plot_confusion_matrix(model, model_name+"_test", test_generator)

def experiment1(train_generator, val_generator, test_generator):
    # In this experiment all layers in the VGG16-model will be initialized with weights from Imagenet
    # and thereafter frozen. The last four fully connected layers in our architecture will be trained.

    # Initialise parameters
    unfrozen_layers = 4
    model_name = "experiment_1" + "_" + dataset

    # Build model
    model = build_model(unfrozen_layers, weights="imagenet")
    print("Model for " + model_name + " built")

    # Train model
    model_statistics = train_model(model, train_generator, val_generator)
    print("Model for " + model_name + " trained")

    # Save model
    if save_model:
        model.save(saved_model_path + model_name + ".h5")
        print("Model for " + model_name + " saved to file")

    # Plot confusion matrix
    if show_confusion_matrix:
        plot_confusion_matrix(model, model_name, val_generator)
        print("Confusion matrix " + model_name + " saved to file")

    # Plot statistics
    if plot_statistics:
        plot_loss_accuracy(model_statistics.history, model_name)
        print("Loss & accuracy plots " + model_name + " saved to file")
    
    # Test model including plotting confusion matrix
    if test_model:
        test_model(model, model_name, test_generator)

def experiment2(train_generator, val_generator, test_generator):
    #  In this experiment all layers in the VGG16-model will be initialized with weights from Imagenet.
    #  All layers will be retrained.

    # Initialise parameters
    unfrozen_layers = 22
    model_name = "experiment_2" + "_" + dataset

    # Build model
    model = build_model(unfrozen_layers, weights="imagenet")
    print("Model for " + model_name + " built")

    # Train model
    model_statistics = train_model(model, train_generator, val_generator)
    print("Model for " + model_name + " trained")

    # Save model
    if save_model:
        model.save(saved_model_path + model_name + ".h5")
        print("Model for " + model_name + " saved to file")

    # Plot confusion matrix
    if show_confusion_matrix:
        plot_confusion_matrix(model, model_name, val_generator)
        print("Confusion matrix " + model_name + " saved to file")

    # Plot statistics
    if plot_statistics:
        plot_loss_accuracy(model_statistics.history, model_name)
        print("Loss & accuracy plots " + model_name + " saved to file")
    
    # Test model including plotting confusion matrix
    if test_model:
        test_model(model, model_name, test_generator)

def experiment3(train_generator, val_generator, test_generator):
    # In this experiment the model will have the same architecture as in the previous experiments,
    # but the weights wont be initialized. All layers will be trained from zero.

    # Initialise parameters
    unfrozen_layers = 22
    model_name = "experiment_3" + "_" + dataset

    # Build model
    model = build_model(unfrozen_layers, weights=None)
    print("Model for " + model_name + " built")

    # Train model
    model_statistics = train_model(model, train_generator, val_generator)
    print("Model for " + model_name + " trained")

    # Save model
    if save_model:
        model.save(saved_model_path + model_name + ".h5")
        print("Model for " + model_name + " saved to file")

    # Plot confusion matrix
    if show_confusion_matrix:
        plot_confusion_matrix(model, model_name, val_generator)
        print("Confusion matrix " + model_name + " saved to file")

    # Plot statistics
    if plot_statistics:
        plot_loss_accuracy(model_statistics.history, model_name)
        print("Loss & accuracy plots " + model_name + " saved to file")

    # Test model including plotting confusion matrix
    if test_model:
        test_model(model, model_name, test_generator)

if __name__ == "__main__":
    # Seed
    random.seed(10)

    # Get generators
    train_generator, val_generator, test_generator = create_generators()

    # Run experiments
    if run_experiment1:
        experiment1(train_generator, val_generator, test_generator)
    if run_experiment2:
        experiment2(train_generator, val_generator, test_generator)
    if run_experiment3:
        experiment3(train_generator, val_generator, test_generator)

    print("All experiment finished!")
