# folders
train_dir = "data/dataset/train/"
save_dir = "/transforms/"

# constants
batch_size = 1

# Transform
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
            rescale=1./255
                )
train_generator = datagen.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=(224, 224),
    save_to_dir=save_dir,
    save_prefix='new',
    save_format='jpeg',
    class_mode='categorical',
    seed=1)

for inputs, outputs in train_generator:
    break