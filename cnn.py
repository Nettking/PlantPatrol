from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the model
model = Sequential()

# Convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Fully connected layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print the model summary
model.summary()
