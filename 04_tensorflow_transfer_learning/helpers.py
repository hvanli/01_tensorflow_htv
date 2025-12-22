from IPython.display import HTML
import tensorflow as tf
import os 
import math
import matplotlib.pyplot as plt
import numpy as np
import keras

def display_tensorboard_logs(url: str = "http://localhost:6006"):
    """
    Display a clickable TensorBoard link inside Jupyter.
    """
    return HTML(f'<a href="{url}" target="_blank">🚀 Open TensorBoard</a>')


IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
LABEL_MODE = "categorical"

def import_and_create_train_test_ds(train_dir, 
                                    test_dir,
                                    label_model=LABEL_MODE,
                                    image_size=IMG_SIZE, 
                                    batch_size=BATCH_SIZE): 
    """
    Imports and creates TensorFlow training and testing datasets from the provided directories.

    Args:
        train_dir (str): Path to the directory containing the training images. The images should be organized into subdirectories, where each subdirectory corresponds to a class label.
        test_dir (str): Path to the directory containing the testing images, organized similarly to the training data.
        image_size (tuple, optional): The target size (height, width) to which each image is resized. Default is (224, 224).
        batch_size (int, optional): The number of images in each batch. Default is 32.

    Returns:
        tuple: A tuple containing two `tf.data.Dataset` objects:
            - train_ds: The training dataset.
            - test_ds: The testing dataset.

    Example:
        train_ds, test_ds = import_and_create_train_test_ds(
            train_dir='/path/to/train',
            test_dir='/path/to/test',
            image_size=(256, 256),
            batch_size=64
        )
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, 
        image_size=image_size, 
        batch_size=batch_size, 
        label_mode=label_model, 
        seed=42
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, 
        image_size=image_size, 
        batch_size=batch_size, 
        label_mode=label_model,
        seed=42
    )

    return train_ds, test_ds

def walk_data_directory(directory):
    for directory, folders, files in os.walk(directory):
        print(f"There are {len(folders)} folders and {len(files)} in '{directory}'")

def view_all_images_in_a_batch_ds(
    data_ds,
    cols=4,                 # fewer columns = bigger images
    img_size=3.5,           # inches per image (increase for bigger)
    max_images=None         # limit images if you want
):
    class_names = getattr(data_ds, "class_names", None)

    images, labels = next(iter(data_ds))

    if max_images:
        images = images[:max_images]
        labels = labels[:max_images]

    n = images.shape[0]
    rows = math.ceil(n / cols)

    figsize = (cols * img_size, rows * img_size)
    plt.figure(figsize=figsize)

    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        if labels.ndim == 1:
            label_idx = int(labels[i].numpy())
        else:
            label_idx = int(np.argmax(labels[i].numpy()))

        title = class_names[label_idx] if class_names else str(label_idx)
        plt.title(title, fontsize=14)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

import datetime

def create_tensorboard_callback(dir_name, experiment_name):

  log_dir = os.path.join(dir_name, experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  print(f"Saving TensorBoard log files to {log_dir}")

  tensorboard_callback = keras.callbacks.TensorBoard(
      log_dir=log_dir,
  )
  return tensorboard_callback


def build_efficientv2_models(variant="B0",
                             num_classes=10, 
                             base_trainable=False, 
                             lr=1e-3,
                             batch_size=32, 
                             dropout=0.2
                            ): 

    model_name = f"EfficientNetV2{variant}"
    try:
        BaseModel = getattr(keras.applications, model_name)
    except AttributeError:
        raise ValueError(f"Invalid EfficientNetV2 variant: {variant}")

    EFFICIENTNETV2_SIZES = {
        "B0": (224, 224),
        "B1": (240, 240),
        "B2": (260, 260),
        "B3": (300, 300),
        "S":  (384, 384),
        "M":  (480, 480),
        "L":  (480, 480),
    }

    img_size = EFFICIENTNETV2_SIZES[variant]

    base_model = BaseModel(
                    include_top=False,
                    weights="imagenet",
                    input_shape=img_size + (3,),
                    pooling="avg",
                )

    base_model.trainable = base_trainable 

    inputs = keras.Input(shape=img_size+(3,), name="input_layer") 
    x = keras.applications.efficientnet_v2.preprocess_input(inputs) 
    x = base_model(x, training=False)
    # x=keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name="output_layer")(x) 

    model = keras.Model(inputs, outputs, name=model_name) 
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(), 
        optimizer=keras.optimizers.Adam(learning_rate=lr), 
        metrics=["accuracy"]
    )
    return model
    
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
  
# Create function to unzip a zipfile into current working directory 
# (since we're going to be downloading and unzipping a few files)
import zipfile

def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results


import math
import numpy as np
import matplotlib.pyplot as plt

def view_same_image_multiple_augmentations(
    data_ds,
    data_augmentation,
    num_images=4,
    num_augments=4,
    img_size=3.5,
    show_original=True,
):
    class_names = getattr(data_ds, "class_names", None)

    images, labels = next(iter(data_ds))
    images = images[:num_images]
    labels = labels[:num_images]

    cols = num_augments + (1 if show_original else 0)
    rows = num_images

    plt.figure(figsize=(cols * img_size, rows * img_size))

    plot_idx = 1
    for i in range(num_images):
        # label
        if labels.ndim == 1:
            label_idx = int(labels[i].numpy())
        else:
            label_idx = int(np.argmax(labels[i].numpy()))
        label_name = class_names[label_idx] if class_names else str(label_idx)

        # original
        if show_original:
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"{label_name}\n(original)", fontsize=12)
            plt.axis("off")
            plot_idx += 1

        # augmented versions (same image, multiple draws)
        for j in range(num_augments):
            plt.subplot(rows, cols, plot_idx)

            aug_img = data_augmentation(images[i], training=True).numpy()

            if aug_img.dtype != np.uint8:
                aug_img = np.clip(aug_img, 0, 255).astype("uint8")

            plt.imshow(aug_img)
            plt.title(f"aug {j+1}", fontsize=11)
            plt.axis("off")
            plot_idx += 1

    plt.tight_layout()
    plt.show()
