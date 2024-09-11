import numpy as np
from keras.utils import to_categorical
def normalize_images(image_list):
    return image_list / 255.0

def convert_labels_to_categorical(labels, num_classes):

    return to_categorical(labels, num_classes=num_classes)
