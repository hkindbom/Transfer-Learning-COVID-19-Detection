""" 
DD2424 Deep Learning in Data Science
May 2020
Group Project: Detecting COVID-19 in X-Rays using Transfer Learning
Author: Mikael Ljung, Hannes Kindbom, Ershard Taherifard, Johanna Dyremark 

Experiment 1: Trains on dataset COVIDxPLUS. All layers in the VGG16-model will be initialized with weights from Imagenet
and thereafter frozen. The last four fully connected layers in our architecture will be trained.

Experiment 2: Trains on dataset COVIDxPLUS. All layers in the VGG16-model will be initialized with weights from Imagenet.
All layers will be retrained.

Experiment 3: Trains on dataset COVIDxMINI. All layers in the VGG16-model will be initialized with weights from Imagenet
and thereafter frozen. The last four fully connected layers in our architecture will be trained.

Experiment 4: Trains on dataset COVIDxMINI. All layers in the VGG16-model will be initialized with weights from Imagenet.
All layers will be retrained.

"""

# Modules
from keras import Model, layers
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import sys

# Initializing experiment booleans
run_experiment1, run_experiment2, run_experiment3, run_experiment4 = False, False, False, False

# Changing chosen experiment boolean to True
exp_id = int(sys.argv[1])
if exp_id == 1: 
    run_experiment1 = True
if exp_id == 2: 
    run_experiment2 = True
if exp_id == 3: 
    run_experiment3 = True
if exp_id == 4: 
    run_experiment4 = True

# Settings of graphics and testing 
show_validation_confusion_matrix = True
plot_statistics = True
evaluate_on_test_data = True

# Datasets, folders and paths
test_dir = 'data/dataset/largeDataset/test/'
saved_accuracy_plot_path = "./accuracy_plot_"
saved_loss_plot_path = "./loss_plot_"
saved_confusion_matrix_path = "./confusion_matrix_"

# Constants
IMG_SIZE = 224 
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
FACTOR = 0.7
PATIENCE = 5
OPTIMIZER = 'Adam'

# Parameters of experiments
if run_experiment1:
    experiment = 'experiment_1'
    unfrozen_layers = 4

if run_experiment2:
    experiment = 'experiment_2'
    unfrozen_layers = 22

if run_experiment3: 
    experiment = 'experiment_3'
    unfrozen_layers = 4

if run_experiment4: 
    experiment = 'experiment_4'
    unfrozen_layers = 22

if run_experiment1 or run_experiment2:
    EPOCHS = 22
    train_dir = './data/dataset/largeDataset/train/'
    val_dir = './data/dataset/largeDataset/val/'
    weights = {0: 1, 1: 1, 2: 12}

if run_experiment3 or run_experiment4:
    EPOCHS = 12
    train_dir = './data/dataset/smallDataset/train/'
    val_dir = './data/dataset/smallDataset/val/'
    weights = 'balanced'


# Reading from folder with training, validation and test data to create ImageDataGenerator object. 
# Augmenting the training data. 
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


# Building the model by creating source VGG-16 model and adding tensors, then freezing layers and compiling
def build_model():
    vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Create our tensors
    last_tensor_vgg = vgg_model.output
    last_tensors = layers.Flatten()(last_tensor_vgg)
    last_tensors = layers.Dense(1024, activation='relu')(last_tensors)
    last_tensors = layers.Dense(1024, activation='relu')(last_tensors)
    last_tensors = layers.Dense(3, activation='softmax')(last_tensors)

    # Add tensors, freeze layers and compile
    model = Model(input=vgg_model.input, output=last_tensors)
    model = freeze_layers(model)
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    return model

# Freeze layers which should not be trained during fine-tuning 
def freeze_layers(model):
    for layer in model.layers[:-unfrozen_layers]:
        layer.trainable = False
    return model

# Train the model using current parameter settings
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

# Plotting the accuracy and loss of train and validation data for each epoch
def plot_loss_accuracy(statistics):
    range_EPOCHS = np.array(range(1, EPOCHS+1))

    plt.plot(range_EPOCHS, statistics['accuracy'])
    plt.plot(range_EPOCHS, statistics['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'Val'], loc='upper left')
    plt.savefig(saved_accuracy_plot_path + experiment + ".png")
    plt.show()

    plt.plot(range_EPOCHS, statistics['loss'])
    plt.plot(range_EPOCHS, statistics['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Val'], loc='upper left')
    plt.savefig(saved_loss_plot_path + experiment + ".png")
    plt.show()

# Plotting the confusion matrix of model's prediction of generator input
def plot_confusion_matrix(model, filename, generator):
    Y_pred = model.predict_generator(generator)
    Y_pred = np.argmax(Y_pred, 1)
    Y_true = generator.classes

    conf_matrix = confusion_matrix(Y_true, Y_pred)
    df_cm = pd.DataFrame(conf_matrix, columns=np.unique(Y_true), index=np.unique(Y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
    plt.savefig(saved_confusion_matrix_path + filename + ".png")
    plt.show()

# Evaluating the accuracy on final test data and plotting the corresponding confusion matrix 
def evaluate_on_test_data(model, test_generator):
    results = model.evaluate_generator(test_generator)
    print(str(model.metrics_names[1]) + " of " + str(experiment) + " on test data: ", results[1])
    plot_confusion_matrix(model, experiment+"_test", test_generator)

# Building and training model, plus plotting conf. matrix, statistics and evaluating model on test data
def run_experiment(train_generator, val_generator, test_generator):

    model = build_model()
    print("Model for " + experiment + " built")

    model_statistics = train_model(model, train_generator, val_generator)
    print("Model for " + experiment + " trained")

    if show_validation_confusion_matrix:
        plot_confusion_matrix(model, experiment+"_validation", val_generator)
        print("Confusion matrix " + experiment + " saved to file")

    if plot_statistics:
        plot_loss_accuracy(model_statistics.history)
        print("Loss & accuracy plots " + experiment + " saved to file")

    if evaluate_on_test_data:
        evaluate_on_test_data(model, test_generator)
        print("Evaluation of " + experiment + "on test data done")

# Get seed, create the generators and run the chosen experiment
if __name__ == "__main__":
    random.seed(10)
    train_generator, val_generator, test_generator = create_generators()
    run_experiment(train_generator, val_generator, test_generator)
