import os
import shutil
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

#data source and parameters


DOWNLOAD_DATA = True
DATA_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
RAW_DATA_DIR = "cell_images"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
TRAIN_SPLIT = 0.8
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42


# Data Acquisition


def download_and_extract_data(download=True):
    if not download:
        print("Skipping data download")
        return
    if os.path.exists(RAW_DATA_DIR):
        shutil.rmtree(RAW_DATA_DIR)
    if os.path.exists("cell_images.zip"):
        os.remove("cell_images.zip")
    os.system(f"wget {DATA_URL}")
    os.system("unzip -q cell_images.zip")
    print("Data downloaded and extracted!")


# Data Processing


def split_train_test():
    random.seed(RANDOM_SEED)
    classes = ['Parasitized', 'Uninfected']

    # Create directories
    for split in ['train', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(f"data/{split}", cls), exist_ok=True)

    # Split and copy files
    for cls in classes:
        class_path = os.path.join(RAW_DATA_DIR, cls)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        train_files, test_files = train_test_split(
            image_files, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED
        )

        for f in train_files:
            shutil.copy2(os.path.join(class_path, f), os.path.join(TRAIN_DIR, cls, f))
        for f in test_files:
            shutil.copy2(os.path.join(class_path, f), os.path.join(TEST_DIR, cls, f))

    print("Train/Test split completed!")


# Create TensorFlow Datasets


def create_tf_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_ds, test_ds

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

def augment_dataset(dataset):
    return dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

if __name__ == "__main__":
    download_and_extract_data(DOWNLOAD_DATA)
    split_train_test()
    train_ds, test_ds = create_tf_datasets()
    train_augmented = augment_dataset(train_ds)
    print("Datasets ready!")