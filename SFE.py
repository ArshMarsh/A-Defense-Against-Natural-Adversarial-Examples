'''
CS475 Project Source Code
Group 11 
'''

from __future__ import print_function, division
import os
import sys
import keras
import numpy as np
import time
from PIL import Image
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras import backend as K
from keras.backend import categorical_crossentropy
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, ZeroPadding2D, MaxPool2D
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from keras.utils import to_categorical
import datetime
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class imagenet_GAN():
    def __init__(self, input_shape, input_latent_dim, G_data, D_data, image_path):
        """

        :param input_shape:
        :param input_latent_dim: the shape input noise of G,should be 1-D array
        :param datasets: the datasets,should be numpy array
        :param image_path: image save path during training
        """
        self.img_shape = input_shape
        self.latent_dim = input_latent_dim
        self.G_datasets = G_data
        self.D_datasets = D_data
        self.image_path = image_path
        self.log = []
        optimizer = tf.keras.optimizers.legacy.Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        frozen_D = Model(
            inputs=self.discriminator.inputs,
            outputs=self.discriminator.outputs)
        frozen_D.trainable = False
        reconstructed_z = self.generator(z)
        validity = frozen_D(reconstructed_z)
        # The discriminator takes generated images as input and determines validity

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        # self.combined = Model(z, validity)
        # self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.combined = Model(z, [reconstructed_z, validity])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Dense(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512, input_dim=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=(self.img_shape,))
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50, rescale=False, expand_dims=True):
        """
        :param epochs: the iteration of training
        :param batch_size: batch_size
        :param sample_interval: print the loss of G and D each sample_interval
        :param rescale: if true,rescale D_img to [-1,1]
        :param expand_dims: if true,expand img channel
        :return:
        """

        # Load the dataset
        D_train = self.D_datasets
        G_train = self.G_datasets

        if rescale:
            # Rescale -1 to 1
            D_train = D_train / 127.5 - 1.
        if expand_dims:
            D_train = np.expand_dims(D_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, D_train.shape[0], batch_size)
            D_imgs = D_train[idx]  # targeted feature
            G_feature = G_train[idx]  # input feature

            noise_add = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = G_feature  # + noise_add

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(D_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, [D_imgs, valid])
            # If at save interval => save generated image samples

            if epoch % sample_interval == 0:
                # Plot the progress
                # message = "%d D loss: %.4f, acc.: %.2f%% G loss: %.4f mse:%.4f r2:%.4f" \
                #           % (epoch, d_loss[0], 100 * d_loss[1], g_loss, mse, r2)
                # self.log.append([epoch, d_loss[0], d_loss[1], g_loss])
                message = "%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1])
                self.log.append([epoch, d_loss[0], d_loss[1], g_loss[0], g_loss[1]])
                self.create_str_to_txt('cnn1', datetime.datetime.now().strftime('%Y-%m-%d'), message)
                print(message)
                # self.sample_images(epoch)

    def showlogs(self, path):
        logs = np.array(self.log)
        names = ["d_loss", "d_acc", "g_loss", "g_mse"]
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.plot(logs[:, 0], logs[:, i + 1])
            plt.xlabel("iteration")
            plt.ylabel(names[i])
            plt.grid()
        plt.tight_layout()
        plt.savefig(path+".png")
        plt.close()
        np.save(path+".npy",logs)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        if not os.path.isdir(self.image_path):
            os.makedirs(self.image_path)
        fig.savefig(self.image_path + "/%d.png" % epoch)
        plt.close()

    def save_model(self, path):
        self.combined.save(path)

    def load_model(self, path):
        self.combined.load_weights(path)

    def get_generator(self):
        return self.generator

    def calculateMSE(self, Y, Y_hat):
        MSE = np.sum(np.power((Y - Y_hat), 2)) / len(Y)
        R2 = 1 - MSE / np.var(Y)
        return MSE, R2

    def create_str_to_txt(self, model_name, date, str_data):
        path_file_name = './{}/imagenet_{}_gan_{}.txt'.format(model_name, model_name, date)
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path_file_name), exist_ok=True)

            # Open file in append mode and write the data
            with open(path_file_name, "a") as f:
                f.write(str_data + '\n')
        except Exception as e:
            # Handle specific exceptions here
            print(f"Error occurred while creating/writing to file: {e}")

def get_sub_model(start_layer_name):
    """
    :param start_layer_name:
    :return: return a sub_model start with the start_layer's input
    """
    start_name = start_layer_name
    new_input = keras.layers.Input(batch_shape=model.get_layer(name=start_layer_name).get_input_shape_at(0))
    print(model.get_layer(name=start_layer_name).get_input_shape_at(0))
    layers_list = [layer.name for layer in model.layers]

    for index, name in enumerate(layers_list):
        if name == start_name:
            sub_list = layers_list[index:]
            break

    for index, sub_layer in enumerate(sub_list):
        if index == 0:
            new_output = model.get_layer(sub_layer)(new_input)
        else:
            new_output = model.get_layer(sub_layer)(new_output)

    sub_model = keras.Model(inputs=new_input, outputs=new_output)
    print(f"Sub_model {sub_list[0]} to {sub_list[-1]}")
    print(f"Sub_model's input is {new_input} and the output is {new_output}")
    return new_input, new_output, sub_model


def scheduler(epoch):
    if epoch <= 80:
        return 0.01
    if epoch <= 140:
        return 0.005
    return 0.001

# Check GPU availability in Colab
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
    print('No GPU found. Please check if GPU is enabled in the notebook settings.')
else:
    print(gpu_info)

# Allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

"""# To Do"""

train_SF_path = ''
train_SF_adv_path = ''
train_NSF_path = ''
train_NSF_adv_path = ''

test_SF_path = ''
test_SF_adv_path = ''
test_NSF_path = ''
test_NSF_adv_path = ''

SF_GAN_log_path = ''
NSF_GAN_log_path = ''
detector_history_path = ''
SF_gan_save_path = ''
NSF_gan_save_path = ''
SF_image_path = ''
NSF_image_path = ''

"""# To Do"""

dataset_dir = ''
adv_dataset_dir = ''

# Get list of classes from directory names
classes = sorted(os.listdir(dataset_dir))
num_classes = len(classes)

# Initialize lists to store images and labels
images = []
labels = []

# Loop through each class folder
for class_index, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip if it's not a directory

    print(f"Loading images from class: {class_name}")

    # Get list of image filenames in the class folder
    image_files = sorted(os.listdir(class_dir))

    # Iterate over image files in the class folder
    for filename in image_files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load image using PIL (assuming images are JPEG or PNG)
            image_path = os.path.join(class_dir, filename)
            image = Image.open(image_path)
            image = image.resize((299, 299))  # Resize if needed
            image = np.array(image)  # Convert PIL image to NumPy array

            # Normalize pixel values to range [0, 1]
            image = image.astype('float32') / 255.0

            # Append image and corresponding label
            images.append(image)
            labels.append(class_index)  # Use class_index as label

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Apply data augmentation to increase the dataset size
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Initialize an empty list to store augmented images and labels
augmented_images = []
augmented_labels = []

# Generate augmented images
for X_batch, y_batch in datagen.flow(images, labels, batch_size=len(images), shuffle=False):
    augmented_images.append(X_batch)
    augmented_labels.append(y_batch)
    if len(augmented_images) >= 4:  # Generate 5 times more data
        break

# Concatenate the augmented data
augmented_images = np.concatenate(augmented_images)
augmented_labels = np.concatenate(augmented_labels)

# Combine original and augmented data
X_combined = np.concatenate([images, augmented_images])
y_combined = np.concatenate([labels, augmented_labels])

# Shuffle combined data
indices = np.arange(len(X_combined))
np.random.shuffle(indices)
X_combined = X_combined[indices]
y_combined = y_combined[indices]

# Split the dataset into training and testing sets
# You can adjust the split ratio as needed
split_ratio = 0.8
split_idx = int(len(X_combined) * split_ratio)

X_train = X_combined[:split_idx]
y_train = y_combined[:split_idx]
X_test = X_combined[split_idx:]
y_test = y_combined[split_idx:]

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

num_classes = 1000  # Assuming ImageNet has 1000 classes
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

classes = sorted(os.listdir(adv_dataset_dir))
num_classes = len(classes)

# Initialize lists to store images and labels
images = []
labels = []

# Loop through each class folder
for class_index, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip if it's not a directory

    print(f"Loading images from class: {class_name}")

    # Get list of image filenames in the class folder
    image_files = sorted(os.listdir(class_dir))

    # Iterate over image files in the class folder
    for filename in image_files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load image using PIL (assuming images are JPEG or PNG)
            image_path = os.path.join(class_dir, filename)
            image = Image.open(image_path)
            image = image.resize((299, 299))  # Resize if needed
            image = np.array(image)  # Convert PIL image to NumPy array

            # Normalize pixel values to range [0, 1]
            image = image.astype('float32') / 255.0

            # Append image and corresponding label
            images.append(image)
            labels.append(class_index)  # Use class_index as label

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Apply data augmentation to increase the dataset size
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Initialize an empty list to store augmented images and labels
augmented_images = []
augmented_labels = []

# Generate augmented images
for X_batch, y_batch in datagen.flow(images, labels, batch_size=len(images), shuffle=False):
    augmented_images.append(X_batch)
    augmented_labels.append(y_batch)
    if len(augmented_images) >= 4:  # Generate 5 times more data
        break

# Concatenate the augmented data
augmented_images = np.concatenate(augmented_images)
augmented_labels = np.concatenate(augmented_labels)

# Combine original and augmented data
X_combined = np.concatenate([images, augmented_images])
y_combined = np.concatenate([labels, augmented_labels])

# Shuffle combined data
indices = np.arange(len(X_combined))
np.random.shuffle(indices)
X_combined = X_combined[indices]
y_combined = y_combined[indices]

# Split the dataset into training and testing sets
# You can adjust the split ratio as needed
split_ratio = 0.8
split_idx = int(len(X_combined) * split_ratio)

adv_data = X_combined[:split_idx]
adv_data_y = y_combined[:split_idx]
adv_test_data = X_combined[split_idx:]
adv_test_data_y = y_combined[split_idx:]

print("Training data shape:", adv_data.shape)
print("Training labels shape:", adv_data_y.shape)
print("Test data shape:", adv_test_data.shape)
print("Test labels shape:", adv_test_data_y.shape)

num_classes = 1000  # Assuming ImageNet has 1000 classes
adv_data_y = to_categorical(adv_data_y, num_classes)
adv_test_data_y = to_categorical(adv_test_data_y, num_classes)

print("Training data shape:", adv_data.shape)
print("Training labels shape:", adv_data_y.shape)
print("Test data shape:", adv_test_data.shape)
print("Test labels shape:", adv_test_data_y.shape)

# Load the InceptionV3 model pre-trained on ImageNet data
model = InceptionV3(weights='imagenet', include_top=True)
model.summary()

# Compile the model with an optimizer and loss function
optimizer = keras.optimizers.Adam()  # Choose your optimizer
loss = keras.losses.CategoricalCrossentropy()  # Choose your loss function
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
loss, accuracy = model.evaluate(adv_data, adv_data_y, verbose=2)
print('adv  loss:%.4f accuracy:%.4f' % (loss, accuracy))
dense1_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)

block = dense1_layer_model.predict(X_train, batch_size=16)
block_adv = dense1_layer_model.predict(adv_data, batch_size=16)
np.save('',block)
np.save('',block_adv)
np.save('',y_train)
print(np.shape(block))
print(np.shape(block_adv))
print(np.shape(y_train))

test_block = dense1_layer_model.predict(X_test, batch_size=16)
test_block_adv = dense1_layer_model.predict(adv_test_data, batch_size=16)

gan_epochs = 1000
gan_batchsize = 16

# train SF_model
targeted_feature = np.concatenate((block, block))
input_block = np.concatenate((block, block_adv))
print("Input block shape:", np.shape(input_block))
print("Targeted feature shape:", np.shape(targeted_feature))
print("\n" * 5)
print("training SF_model")
time_start = time.time()
gan = imagenet_GAN(input_shape=2048, input_latent_dim=2048, G_data=input_block, D_data=targeted_feature,
                image_path=SF_image_path)

gan.train(epochs=gan_epochs, batch_size=gan_batchsize, sample_interval=200, rescale=False, expand_dims=False)

gan.showlogs(path=SF_GAN_log_path)
model_save_path = '/content/drive/MyDrive/CS475_Project_Code/try_1/SF_Model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
gan.save_model(model_save_path + SF_gan_save_path)
train_SF_pre = gan.generator.predict(block, batch_size=6)
train_SF_adv_pre = gan.generator.predict(block_adv, batch_size=6)
time_end = time.time()
print('totally cost', time_end - time_start)
np.save(train_SF_path, train_SF_pre)
np.save(train_SF_adv_path, train_SF_adv_pre)

test_SF_pre = gan.generator.predict(test_block, batch_size=6)
test_SF_adv_pre = gan.generator.predict(test_block_adv, batch_size=6)
np.save(test_SF_path, test_SF_pre)
np.save(test_SF_adv_path, test_SF_adv_pre)

print("\n" * 5)
print("training NSF_model")
# train NSF_model
targeted_feature = np.concatenate((block_adv, block_adv))
input_block = np.concatenate((block, block_adv))
print(np.shape(input_block), np.shape(targeted_feature))

gan = imagenet_GAN(input_shape=2048, input_latent_dim=2048, G_data=input_block, D_data=targeted_feature,
                image_path=NSF_image_path)
gan.train(epochs=gan_epochs, batch_size=gan_batchsize, sample_interval=200, rescale=False, expand_dims=False)
gan.showlogs(path=NSF_GAN_log_path)
model_save_path = '/content/drive/MyDrive/CS475_Project_Code/try_1/NSF_Model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
gan.save_model(model_save_path + NSF_gan_save_path)
train_NSF_pre = gan.generator.predict(block, batch_size=6)
train_NSF_adv_pre = gan.generator.predict(block_adv, batch_size=6)
np.save(train_NSF_path, train_NSF_pre)
np.save(train_NSF_adv_path, train_NSF_adv_pre)

test_NSF_pre = gan.generator.predict(test_block, batch_size=6)
test_NSF_adv_pre = gan.generator.predict(test_block_adv, batch_size=6)
np.save(test_NSF_path, test_NSF_pre)
np.save(test_NSF_adv_path, test_NSF_adv_pre)

# testing the acc based on ori_model
new_input, new_output, sub_model = get_sub_model('predictions')
sub_model.summary()
sgd = keras.optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
sub_model.compile(loss=categorical_crossentropy, optimizer=sgd,
                  metrics=['accuracy'])
print(sub_model.input)
print("-"*5, "evaluate train data","-"*5)
loss, accuracy = sub_model.evaluate(block, adv_data_y, verbose=2)
print('SF train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(train_SF_pre, adv_data_y, verbose=2)
print('SF pre_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(block_adv, adv_data_y, verbose=2)
print('SF adv_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(train_SF_adv_pre, adv_data_y, verbose=2)
print('SF pre_adv_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))

loss, accuracy = sub_model.evaluate(block, adv_data_y, verbose=2)
print('NSF train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(train_NSF_pre, adv_data_y, verbose=2)
print('NSF pre_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(block_adv, adv_data_y, verbose=2)
print('NSF adv_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(train_NSF_adv_pre, adv_data_y, verbose=2)
print('NSF pre_adv_train  loss:%.4f accuracy:%.4f' % (loss, accuracy))

print("-" * 5, "evaluate test data", "-" * 5)
loss, accuracy = sub_model.evaluate(test_block, adv_test_data_y, verbose=2)
print('SF test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(test_SF_pre, adv_test_data_y, verbose=2)
print('SF pre_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(test_block_adv, adv_test_data_y, verbose=2)
print('SF adv_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(test_SF_adv_pre, adv_test_data_y, verbose=2)
print('SF pre_adv_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))

loss, accuracy = sub_model.evaluate(test_block, adv_test_data_y, verbose=2)
print('NSF test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(test_NSF_pre, adv_test_data_y, verbose=2)
print('NSF pre_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(test_block_adv, adv_test_data_y, verbose=2)
print('NSF adv_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))
loss, accuracy = sub_model.evaluate(test_NSF_adv_pre, adv_test_data_y, verbose=2)
print('NSF pre_adv_test  loss:%.4f accuracy:%.4f' % (loss, accuracy))

def load_feature(attack_name):
  train_SF_pre = np.load(train_SF_path)
  train_SF_adv_pre = np.load(train_SF_adv_path)
  train_NSF_pre = np.load(train_NSF_path)
  train_NSF_adv_pre = np.load(train_NSF_adv_path)

  test_SF_pre = np.load(test_SF_path)
  test_SF_adv_pre = np.load(test_SF_adv_path)
  test_NSF_pre = np.load(test_NSF_path)
  test_NSF_adv_pre = np.load(test_NSF_adv_path)

  # train datasets
  train_ori_SF_and_NSF = np.concatenate((train_SF_pre, train_SF_adv_pre), axis=1)
  train_ori_label = np.zeros(shape=(len(train_ori_SF_and_NSF), 1))
  train_adv_SF_and_NSF = np.concatenate((train_NSF_pre, train_NSF_adv_pre), axis=1)
  train_adv_label = np.ones(shape=(len(train_adv_SF_and_NSF), 1))
  print(np.shape(train_ori_SF_and_NSF), np.shape(train_adv_label))
  train_SF_and_NSF = np.concatenate((train_ori_SF_and_NSF, train_adv_SF_and_NSF))
  train_SF_and_NSF_label = np.concatenate((train_ori_label, train_adv_label))

  # test datasets
  test_ori_SF_and_NSF = np.concatenate((test_SF_pre, test_SF_adv_pre), axis=1)
  test_ori_label = np.zeros(shape=(len(test_ori_SF_and_NSF), 1))
  test_adv_SF_and_NSF = np.concatenate((test_NSF_pre, test_NSF_adv_pre), axis=1)
  test_adv_label = np.ones(shape=(len(test_adv_SF_and_NSF), 1))
  print(np.shape(test_ori_SF_and_NSF), np.shape(test_ori_label))
  test_SF_and_NSF = np.concatenate((test_ori_SF_and_NSF, test_adv_SF_and_NSF))
  test_SF_and_NSF_label = np.concatenate((test_ori_label, test_adv_label))

  # SF_and_NSF_label = keras.utils.to_categorical(SF_and_NSF_label,num_classes=2)
  print("train:", train_SF_and_NSF_label[0], train_SF_and_NSF_label.shape, train_SF_and_NSF.shape)
  print("test:", test_SF_and_NSF_label[0], test_SF_and_NSF_label.shape, test_SF_and_NSF.shape)
  print("-" * 10, attack_name, "-" * 10)

  train_x = train_SF_and_NSF
  train_y = train_SF_and_NSF_label
  test_x = test_SF_and_NSF
  test_y = test_SF_and_NSF_label


  return train_x,train_y,test_x,test_y

def MLP(dropout_rate=0.25, activation='relu',classes=1):
  start_neurons = 512
  model = Sequential()
  model.add(Dense(start_neurons, input_dim=256, activation=activation))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))

  model.add(Dense(start_neurons // 2, activation=activation))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))

  model.add(Dense(start_neurons // 4, activation=activation))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))

  model.add(Dense(start_neurons // 8, activation=activation))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate / 2))

  model.add(Dense(classes, activation='sigmoid'))
  return model

def plot_loss_acc(history, fold, base_path,acc,max_epoch):
  history_dict = history.history
  history_dict.keys()
  loss_values = history_dict['loss']
  val_loss_values = history_dict['val_loss']
  epoch = range(1, len(loss_values) + 1)
  plt.plot(epoch, loss_values, label='Training loss')
  plt.plot(epoch, val_loss_values, label='Validation loss')
  plt.title("Training and validation loss at acc:%.2f%%" % acc)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  my_x_ticks = np.arange(0, max_epoch+1,1 )
  plt.xticks(my_x_ticks)
  plt.legend()
  plt.savefig(base_path + "_fold_" + str(fold) + "_loss.png")
  plt.close()

  acc_values = history_dict['acc']
  val_acc_values = history_dict['val_acc']
  plt.plot(epoch, acc_values, label='Training acc')
  plt.plot(epoch, val_acc_values, label='Validation acc')
  plt.title("Training and validation accuracy at acc:%.2f%%" % acc)
  plt.xlabel('Epochs')
  my_x_ticks = np.arange(0, max_epoch+1, 1)
  plt.xticks(my_x_ticks)
  plt.ylabel('Acc')
  plt.legend()
  plt.savefig(base_path + "_fold_" + str(fold) + "_acc.png")
  plt.close()
  np.save(base_path + "_fold_" + str(fold) + ".npy", history_dict)

"""# To Do"""

detector_history_path = '/content/drive/MyDrive/CS475_Project_Code/try_1'
model_save_path = '/content/drive/MyDrive/CS475_Project_Code/try_1'

attack_name = 'natural_adversarial'
detect_name = 'natural_adversarial'

train_x,train_y,test_x,test_y = load_feature(attack_name)
print(train_y[0],train_y[10000])
classes = 1

index = np.arange(len(train_x[0:10000]))
np.random.shuffle(index)
train_x = train_x[index]
train_y = train_y[index]

index = np.arange(len(test_x[0:2000]))
np.random.shuffle(index)
test_x = test_x[index]
test_y = test_y[index]

K.clear_session()
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

folds = KFold(n_splits=5, shuffle=True, random_state=2019)

patience = 10  # How many steps to stop
call_ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1,
                                        mode='auto', baseline=None)
epochs = 25
batch_size = 256
cvscores_train = []
cvscores_test = []
checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_acc', verbose=1,
                                                  save_best_only=True, mode='max')
model = MLP(dropout_rate=0.5, activation='relu')
sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights(model_save_path)
scores = model.evaluate(train_x[0:10000], train_y[0:10000], verbose=2)
print("train_sub_val %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
cvscores_train.append(scores[1] * 100)
scores = model.evaluate(test_x[0:2000], test_y[0:2000], verbose=2)
print("test %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

time_start = time.time()
history = model.fit(train_x[0:10000], train_y[0:10000],
                    # validation_data=[test_x, test_y],
                    validation_split =0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[call_ES,checkpoint ],
                    shuffle=True,
                    verbose=1)
time_end = time.time()
print('totally cost', time_end - time_start)
time_start = time.time()
scores = model.evaluate(train_x[0:10000], train_y[0:10000], verbose=2)
time_end = time.time()
print('totally cost', time_end - time_start)
print("train_sub_val %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
cvscores_train.append(scores[1] * 100)
scores = model.evaluate(test_x[0:2000], test_y[0:2000], verbose=2)
print("test %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
cvscores_test.append(scores[1] * 100)
plot_loss_acc(history, 0, detector_history_path, scores[1]*100,max_epoch=epochs)
print("-" * 10, attack_name, "-" * 10)