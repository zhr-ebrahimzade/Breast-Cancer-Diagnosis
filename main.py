from data_preprocessing import load_data, preprocess_images, random_selecting
from utils import normalize_images, convert_labels_to_categorical
from model_training import build_model, train_model
import tensorflow as tf
import keras
import numpy as np

# File paths
xlsx_path = '/content/drive/MyDrive/Colab Notebooks/Radiology manual annotations.xlsx'
image_folder_path = "/content/drive/MyDrive/Colab Notebooks/Subtracted images of CDD-CESM"

# Load and preprocess data
df = load_data(xlsx_path)
images = preprocess_images(image_folder_path, df.Image_name.values)

# Split data
validation_data_image, validation_dataFrame, remaining_image_list, remaining_df = random_selecting(images, df, 0.14)
test_data_image, test_dataFrame, remaining_image_list, remaining_df = random_selecting(remaining_image_list, remaining_df, 0.12)
training_set, training_dataFrame, _, _ = random_selecting(remaining_image_list, remaining_df, 1)

# Normalize images
validation_data_image = normalize_images(validation_data_image)
test_data_image = normalize_images(test_data_image)
training_set = normalize_images(training_set)

# Convert labels to categorical
labels = {'Normal': 0, 'Benign': 1, 'Malignant': 2}
train_labels = convert_labels_to_categorical(training_dataFrame['Titles'].map(labels), num_classes=len(labels))
val_labels = convert_labels_to_categorical(validation_dataFrame['Titles'].map(labels), num_classes=len(labels))

# Build and train the model
model = build_model(input_shape=(224, 224, 3), num_classes=len(labels))

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath='/content/drive/MyDrive/bestModel/BestModel.h5', monitor='val_accuracy', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
]

train_model(model, training_set, train_labels, (validation_data_image, val_labels), callbacks)

print("Training complete.")
