from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, layers
from keras.callbacks import ReduceLROnPlateau

# folders
train_dir = 'data/dataset/train/'
val_dir = 'data/dataset/val/'
test_dir = 'data/dataset/test/'

# constants
IMG_SIZE = 224  # double check in report
LEARNING_RATE = 2e-5
EPOCHS = 3  # in report 22
BATCH_SIZE = 1
FACTOR = 0.7
PATIENCE = 5
OPTIMIZER = 'Adam'


def create_generators():
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='constant',
        cval=0.,
        rescale=1. / 255
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        batch_size=BATCH_SIZE,
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='categorical')

    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

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
    model = freeze_layers(model, unfrozen_layers=4)

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

    trained_model = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=nb_val_samples // BATCH_SIZE,
        verbose=0,
        callbacks=[ReduceLROnPlateau(
            monitor='loss',
            factor=FACTOR,
            patience=PATIENCE,
            verbose=0)])

    return trained_model


if __name__ == "__main__":
    # Get generators
    train_generator, val_generator, test_generator = create_generators()

    # Build model
    model = build_model(unfrozen_layers=4)
    model.summary()

    # Train model
    trained_model = train_model(model, train_generator, val_generator)
    print("Model trained")

    # evaluate model
    result = trained_model.evaluate_generator(test_generator)
    print("Model evaluated")
