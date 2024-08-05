import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.25, random_state=42
)

# Define the CNN model
model = Sequential([
    tf.keras.layers.InputLayer(shape=(28, 28, 1)),
    Conv2D(8, (3, 3), padding='same'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D((2, 2), strides=2),

    Conv2D(16, (3, 3), padding='same'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D((2, 2), strides=2),

    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    ReLU(),

    Flatten(),
    Dense(10),
    Softmax()
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define training options
epochs = 4
batch_size = 32
validation_data = (val_images, val_labels)

# Train the model
history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                    validation_data=validation_data, verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Predict labels for the validation set
val_predictions = model.predict(val_images)
val_predicted_labels = tf.argmax(val_predictions, axis=1)
val_true_labels = tf.argmax(val_labels, axis=1)

# Calculate accuracy on validation set
val_accuracy = tf.reduce_mean(tf.cast(val_predicted_labels == val_true_labels, tf.float32))
print(f'Validation accuracy: {val_accuracy * 100:.2f}%')
