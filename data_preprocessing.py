import numpy as np
import pandas as pd
import cv2
import os

def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df[df['Type'] == 'CESM']
    df = df.rename(columns={"Pathology Classification/ Follow up": "Titles"})
    df = df.drop_duplicates()
    df["Image_name"] = df["Image_name"].str.strip()
    df = df.sort_values(by=["Image_name"])
    return df

def preprocess_images(image_folder_path, image_names):
    image_files = [os.path.join(image_folder_path, f"{name}.jpg") for name in image_names]
    resized_images = []
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image, (224, 224))
            resized_images.append(resized_image)
        else:
            print(f"Image not found: {image_path}")
    return np.array(resized_images)

def random_selecting(image_list, df, percentage):
    data_size = int(len(image_list) * percentage)
    random_select = np.random.choice(len(image_list), size=data_size, replace=False)
    data_image = np.array(image_list)[random_select]
    data_df = df.iloc[random_select].reset_index(drop=True)
    remaining_image_list = np.delete(image_list, random_select, axis=0)
    remaining_df = df.drop(df.index[random_select]).reset_index(drop=True)
    return data_image, data_df, remaining_image_list, remaining_df
