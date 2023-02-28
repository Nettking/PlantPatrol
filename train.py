from keras.preprocessing.image import ImageDataGenerator

# Define the train and validation data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load the train and validation data
train_generator = train_datagen.flow_from_directory('path/to/train',
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory('path/to/val',
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='categorical')

# Train the model
model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.n // train_generator.batch_size,
                    epochs=10,
                    validation_data=val_generator,
                    validation_steps=val_generator.n // val_generator.batch_size)
