import os

TRAIN_PATH = "train/"
TEST_PATH = "test/"

if not os.path.exists(TRAIN_PATH + "0"):
    os.mkdir(TRAIN_PATH + "0")
if not os.path.exists(TRAIN_PATH + "1"):
    os.mkdir(TRAIN_PATH + "1")
if not os.path.exists(TEST_PATH + "0"):
    os.mkdir(TEST_PATH + "0")
if not os.path.exists(TEST_PATH + "1"):
    os.mkdir(TEST_PATH + "1")

train_images = [f for f in os.listdir(TRAIN_PATH) if f.endswith("jpg")]
test_images = [f for f in os.listdir(TEST_PATH) if f.endswith("jpg")]

for image in train_images:
    name, label = image.split("_")
    label = label[0]
    os.rename(TRAIN_PATH + image, TRAIN_PATH + f"{label}/" + name + ".jpg")

for image in test_images:
    name, label = image.split("_")
    label = label[0]
    os.rename(TEST_PATH + image, TEST_PATH + f"{label}/" + name + ".jpg")