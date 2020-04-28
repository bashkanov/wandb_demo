import os
import wandb
wandb.init(project="classification_with_wandb")
# os.environ["WANDB_MODE"] = "dryrun"

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD

tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

config = wandb.config
model_path = os.path.join(wandb.run.dir, "class_model.h5")

# Data utils
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    image = tf.image.resize(image, [32, 32])
    return image, label

def augment_data(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 36, 36)
    image = tf.image.random_crop(image, size=[32, 32, 1])
    image = tf.image.random_brightness(image, max_delta=0.5)
    return image, label

def get_batched_dataset(data, batch_size, augment=False, shufbuf=500):
    data = tf.data.Dataset.from_tensor_slices(data)
    if shufbuf:
        data = data.shuffle(shufbuf)
    data = data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if augment:
        data = data.map(augment_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=True)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data


# Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
train_dataset, test_dataset = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = train_dataset, test_dataset


start_features = config.start_features
model = keras.Sequential([
    keras.layers.Input(shape=[32, 32, 1]),
    keras.layers.Conv2D(start_features, [3, 3], padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(),

    keras.layers.Conv2D(start_features*2, [3, 3], padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(),

    keras.layers.Conv2D(start_features*4, [3, 3], padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(),

    keras.layers.Conv2D(start_features*8, [3, 3], padding='same'),
    keras.layers.Activation('relu'),

    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

optimizer_us = eval(config.optimizer)(lr=float(config.learning_rate))
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# start logging with training
train_loss, valid_loss = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()   
train_acc, valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseCategoricalAccuracy()


# Train the model
@tf.function
def model_train(features, labels, model, optimizer):
    # Define the GradientTape context
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update the loss and accuracy
    train_loss(loss)
    train_acc(labels, predictions)


# Validating the model
@tf.function
def model_validate_us(features, labels, model):
    predictions = model(features)
    v_loss = loss_func(labels, predictions)
    valid_loss(v_loss)
    valid_acc(labels, predictions)


train_dataset = get_batched_dataset(train_dataset, config.batch_size, augment=config.data_augmentation)
test_dataset = get_batched_dataset(test_dataset, config.batch_size)

epochs = config.epochs
for epoch in range(epochs):
    wandb.run.summary["run_id"] = os.path.basename(wandb.run.dir)    
    # Training
    for i, (img, lbl) in enumerate(train_dataset):
        model_train(img, lbl, model, optimizer_us)
    
    # Validate
    for i, (img, lbl) in enumerate(test_dataset):
        model_validate_us(img, lbl, model)

    model.save(model_path, overwrite=True)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    if epoch % 1 == 0:
        print(template.format(epoch + 1,
                              round(train_loss.result().numpy()*1, 2),
                              round(train_acc.result().numpy() * 100, 2),
                              round(valid_loss.result().numpy(), 2),
                              round(valid_acc.result().numpy() * 100, 2)))
        
        wandb.log(dict(loss=round(train_loss.result().numpy(), 2),
                       accuracy=round(train_acc.result().numpy() * 100, 2),
                       test_loss=round(valid_loss.result().numpy(), 2),
                       test_accuracy=round(valid_acc.result().numpy() * 100, 2)), 
                  step=epoch)

    train_loss.reset_states(), 
    valid_loss.reset_states()
    train_acc.reset_states()
    valid_acc.reset_states()