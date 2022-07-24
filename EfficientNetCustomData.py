import os
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetB0
from keras.applications import EfficientNetB1
from keras.applications import EfficientNetB2
from keras.applications import EfficientNetB3
from keras.applications import EfficientNetB4
from keras.applications import EfficientNetB5
from keras.applications import EfficientNetB6
from keras.applications import EfficientNetB7
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    #plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


class EfficientType(Enum):
    B0 = 224
    B1 = 240
    B2 = 260
    B3 = 300
    B4 = 380
    B5 = 456
    B6 = 528
    B7 = 600


class EfficientNetTrainer:

    dataset_path = ""
    dataset_path_str = ""
    model_name = ""

    num_classes = 0
    img_size = 0
    batch_size = 0
    epochs = 0

    class_labels = []
    images = []
    labels = []

    data_frame = None

    encoded_labels = None

    train_x = test_x = train_y = test_y = None

    model = None

    efficient_type = None

    with_eager = None

    def __init__(self, model_name, efficient_type, dataset_path):
        self.model_name = model_name
        self.efficient_type = efficient_type
        self.dataset_path = os.listdir(dataset_path)
        self.img_size = self.efficient_type.value
        self.dataset_path_str = dataset_path
        print(self.dataset_path)

    def initialize(self, batch_size=32, epochs=12, ):
        print("Types of classes labels found: ", len(self.dataset_path))
        self.num_classes = len(self.dataset_path)
        self.img_size = self.efficient_type.value
        self.efficient_type = self.efficient_type
        self.batch_size = batch_size
        self.epochs = epochs

    def automized_training(self, with_check, batch_size, epochs, with_eager=False, scratch=False, weight="imagenet"):
        self.with_eager = with_eager  # set eager
        self.initialize(batch_size, epochs)  # calling initialize and set main data
        self.build_data_frame(with_check)  # build data frame with labels
        self.create_image_set()  # create the image set (resize, orderd labels)
        self.transform_labels()  # encode label
        self.split_set(with_check)  # split set in train and test data

        self.pre_setup()  # limit gpu, set eager of or on

        if scratch:
            self.create_efficientNetModel_scratch()  # model without weights
        else:
            self.create_weighted_model(weight)  # model with weight

        self.train_and_save_model()

    def build_data_frame(self, with_check):
        for item in self.dataset_path:
            all_classes = os.listdir(self.dataset_path_str + "/" + item)
            for room in all_classes:
                self.class_labels.append((item, str('dataset_path' + "/" + item) + '/' + room))
        self.data_frame = pd.DataFrame(data=self.class_labels, columns=['Labels', 'image'])
        if with_check:
            self.check_data_frame()

    def check_data_frame(self):
        print((self.data_frame.head()))
        print(self.data_frame.tail())
        print("Total number of images in the dataset: ", len(self.data_frame))

        label_count = self.data_frame['Labels'].value_counts()
        print(label_count)

    def create_image_set(self):
        for i in self.dataset_path:
            data_set_path = self.dataset_path_str + '/' + str(i)
            filenames = [i for i in os.listdir(data_set_path)]

            for f in filenames:
                data_path = data_set_path + "/" + f
                img = cv2.imread(data_path)
                try:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                except:
                    print(data_path)
                    os.remove(data_path)

                self.images.append(img)
                self.labels.append(i)

        self.images = np.array(self.images)

        try:
            self.images = self.images.astype('float32') / 255
        except:
            print(data_path)
            os.remove(data_path)

        print("TrainingsSize, Width, Height, Channel")
        print(self.images.shape)

    def transform_labels(self):
        self.encoded_labels = self.data_frame['Labels'].values
        print(self.encoded_labels)

        y_label_encoder = LabelEncoder()
        self.encoded_labels = y_label_encoder.fit_transform(self.encoded_labels)
        print(self.encoded_labels)

        self.encoded_labels = self.encoded_labels.reshape(-1, 1)

        ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
        self.encoded_labels = ct.fit_transform(self.encoded_labels)

        print(self.encoded_labels[:5])
        print(self.encoded_labels[35:])

    def split_set(self, with_check):
        self.images, self.encoded_labels = shuffle(self.images, self.encoded_labels, random_state=1)

        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.images, self.encoded_labels, test_size=0.1, random_state=415)

        if with_check:
            print("Train X")
            print(self.train_x)

            print("Test X")
            print(self.test_x)

            print("Train Y")
            print(self.train_y)

            print("Test Y")
            print(self.test_y)

    def getKerasEfficientNet(self, inputs):

        include_top = True
        weights = None

        if self.efficient_type == EfficientType.B0:
            return EfficientNetB0(include_top=include_top, weights=weights, classes=self.num_classes)(inputs)
        elif self.efficient_type == EfficientType.B1:
            return EfficientNetB1(include_top=include_top, weights=weights, classes=self.num_classes)(inputs)
        elif self.efficient_type == EfficientType.B2:
            return EfficientNetB2(include_top=include_top, weights=weights, classes=self.num_classes)(inputs)
        elif self.efficient_type == EfficientType.B3:
            return EfficientNetB3(include_top=include_top, weights=weights, classes=self.num_classes)(inputs)
        elif self.efficient_type == EfficientType.B4:
            return EfficientNetB4(include_top=include_top, weights=weights, classes=self.num_classes)(inputs)
        elif self.efficient_type == EfficientType.B5:
            return EfficientNetB5(include_top=include_top, weights=weights, classes=self.num_classes)(inputs)
        elif self.efficient_type == EfficientType.B6:
            return EfficientNetB6(include_top=include_top, weights=weights, classes=self.num_classes)(inputs)
        elif self.efficient_type == EfficientType.B7:
            return EfficientNetB7(include_top=include_top, weights=weights, classes=self.num_classes)(inputs)

    def get_weighted_Keras(self,weights):

        include_top = False

        if self.efficient_type == EfficientType.B0:
            return EfficientNetB0(include_top=include_top, weights=weights, classes=self.num_classes,
                                  input_shape=(self.img_size, self.img_size, 3))
        elif self.efficient_type == EfficientType.B1:
            return EfficientNetB1(include_top=include_top, weights=weights, classes=self.num_classes,
                                  input_shape=(self.img_size, self.img_size, 3))
        elif self.efficient_type == EfficientType.B2:
            return EfficientNetB1(include_top=include_top, weights=weights, classes=self.num_classes,
                                  input_shape=(self.img_size, self.img_size, 3))
        elif self.efficient_type == EfficientType.B3:
            return EfficientNetB1(include_top=include_top, weights=weights, classes=self.num_classes,
                                  input_shape=(self.img_size, self.img_size, 3))
        elif self.efficient_type == EfficientType.B4:
            return EfficientNetB1(include_top=include_top, weights=weights, classes=self.num_classes,
                                  input_shape=(self.img_size, self.img_size, 3))
        elif self.efficient_type == EfficientType.B5:
            return EfficientNetB1(include_top=include_top, weights=weights, classes=self.num_classes,
                                  input_shape=(self.img_size, self.img_size, 3))
        elif self.efficient_type == EfficientType.B6:
            return EfficientNetB1(include_top=include_top, weights=weights, classes=self.num_classes,
                                  input_shape=(self.img_size, self.img_size, 3))
        elif self.efficient_type == EfficientType.B7:
            return EfficientNetB1(include_top=include_top, weights=weights, classes=self.num_classes,
                                  input_shape=(self.img_size, self.img_size, 3))

    def pre_setup(self):
        max_memory = 6000  # dedicated memory in MB; run 'dxdiag' to get exact figure
        max_usage = 0.75 * max_memory  # example for using up to 95%

        if not self.with_eager:
            tf.compat.v1.disable_eager_execution()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_usage)])

    def create_efficientNetModel_scratch(self):

        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        outputs = self.getKerasEfficientNet(inputs)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.summary()

    def create_weighted_model(self, weight):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        # x = img_augmentation(inputs)
        x = inputs
        self.model = EfficientNetB3(include_top=False, input_tensor=x, weights=weight)

        # Freeze the pretrained weights
        self.model.trainable = False

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(self.model.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = 0.8
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="pred")(x)

        # Compile
        self.model = tf.keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train_and_save_model(self):
        hist = self.model.fit(self.train_x, self.train_y, epochs=self.epochs, verbose=1, batch_size=self.batch_size)

        preds = self.model.evaluate(self.test_x, self.test_y)
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))

        self.model.save(self.model_name)

        plot_hist(hist)

    def create_img_tensor(self, img_path):
        print(img_path)
        img = image.image_utils.load_img(img_path, target_size=(self.img_size, self.img_size))
        img_tensor = image.image_utils.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255
        return img_tensor

    def test_model(self, limit, *args):
        """
        @args takes various image paths, like = "Directory/image1.jpg"
        Will print the name of the image and the 3 best predictions
        """
        self.model = tf.keras.models.load_model(self.model_name)

        def pretty_print(x):
            return np.format_float_positional(x, trim="-")

        for img_path in args:

            img_tensor = self.create_img_tensor(img_path)
            prediction = self.model.predict(img_tensor)
            prediction = prediction.ravel()


            result = zip(prediction, self.dataset_path)
            result = sorted(result)
            print(img_path + ":")
            for key, value in result:
                pass
                print(value + "=" + pretty_print(key))
            print("-----------------------------------------------------")

    def convert_h5_to_tflite(self, model_name):
        self.model = tf.keras.models.load_model(model_name)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)


def create_test_img_paths(directory):
    images = os.listdir(directory)
    return [directory + "/" + images[x] for x in range(len(images))]


def create_test_img_paths_with_index(directory, index):
    images = os.listdir(directory)
    return [directory + "/" + images[x] for x in range(index)]


trainer = EfficientNetTrainer(dataset_path="Bilder", efficient_type=EfficientType.B3, model_name="bilder.h5")
#trainer.automized_training(with_check=True, epochs=11, batch_size=32, with_eager=False, scratch=False)

trainer.test_model(False, *create_test_img_paths("TestPck"))

# trainer.test_model("TestPck/hund.jpg")
# trainer.test_model_single_picture("TestPck/Katza.jpg")
# trainer.test_model("TestPck/Katza2.jpg")
# trainer.test_model("TestPck/Rico.jpg")
# trainer.test_model("TestPck/Fee.jpg")
# trainer.test_model("TestPck/Vanessa2.jpg")
# trainer.test_model("Bilderx3/1_Portrait/2.jpg")


# trainer.convert_h5_to_tflite()
