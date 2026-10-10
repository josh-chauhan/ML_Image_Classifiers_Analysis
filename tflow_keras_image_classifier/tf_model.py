import os
import time

# Quiet TensorFlow's startup logging (must be set before importing tensorflow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

layers = keras.layers

# Set to False to save plots to disk without opening windows (windows block the script until closed)
SHOW_PLOTS = True

# Makes results reproducible (seeds Python, NumP y, and TensorFlow at once)
keras.utils.set_random_seed(42)

print("TensorFlow", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
# TensorFlow did not support the GPU, so on my system the speed comparison won't be fair
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"


def finish_plot(filename):
    """Save the current figure, then show or close it depending on SHOW_PLOTS."""
    plt.savefig(filename)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# Load CIFAR-10 (downloads on first run, then cached in ~/.keras/datasets)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()  # labels come as shape (N, 1), flatten to (N,)
y_test = y_test.flatten()

# Same preprocessing as the PyTorch version: scale to 0-1, then normalize to -1..1
x_train = (x_train.astype("float32") / 255.0 - 0.5) / 0.5
x_test = (x_test.astype("float32") / 255.0 - 0.5) / 0.5

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Preview the first 10 training images
fig, axes = plt.subplots(1, 10, figsize=(12, 3))
for i in range(10):
    denormalized_image = x_train[i] / 2 + 0.5  # matplotlib requires pixel values to be 0-1
    axes[i].imshow(denormalized_image)
    axes[i].set_title(classes[y_train[i]])
    axes[i].axis('off')
finish_plot('sample_images_tf.png')

# Same architecture as the PyTorch CNN (639,742 trainable parameters)
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),

    # Convolution + max pooling layers
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2, strides=2),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(2, strides=2),

    # Flatten feature map, to be fed into dense layers
    layers.Flatten(),

    # Fully connected layers
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.summary()

# Adam with lr=0.001, same as PyTorch. Sparse categorical cross-entropy on softmax outputs
# is equivalent to PyTorch's log_softmax + NLLLoss.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

epochs = 10
batch_size = 32

start_time = time.time()
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2,  # one line per epoch
)
training_time = time.time() - start_time
print(f'Finished Training in {training_time:.1f}s')

# Reusable model
model.save('cifar_cnn_tf.keras')

# Keras records the average training loss per epoch automatically
epoch_losses = history.history['loss']

# Plot the training loss curve
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average training loss')
plt.title('TensorFlow/Keras CNN Training Loss')
plt.grid(True)
plt.tight_layout()
finish_plot('training_loss_tf.png')


def view_result(image, probabilities, actual_label):
    probabilities = np.squeeze(probabilities)
    predicted_label = int(probabilities.argmax())

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)

    denormalized_image = image / 2 + 0.5  # Keras images are already height x width x channels
    ax1.imshow(denormalized_image)
    ax1.set_title(f'True: {classes[actual_label]}\nPredicted: {classes[predicted_label]}')
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


# Predict on one test image (the model already outputs probabilities, no exp() needed)
sample_image = x_test[0]
sample_label = int(y_test[0])
probabilities = model.predict(sample_image[np.newaxis, ...], verbose=0)  # add batch dimension

view_result(sample_image, probabilities, sample_label)
finish_plot('prediction_tf.png')

# Measure accuracy on the full test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

print(f'Test accuracy on the {len(x_test)} test images: {100 * test_accuracy:.2f}%')
print(f'Training time: {training_time:.1f}s on {device_name}')