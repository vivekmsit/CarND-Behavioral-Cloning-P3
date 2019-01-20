import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

data_directory = '../data/'
#data_directory = '/opt/carnd_p3/data/'

def plot_loss_data(history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys())
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('examples/loss_data.png')
    #plt.show()

def get_sample_data():
    lines = []
    with open(data_directory + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip header
        for line in reader:
            lines.append(line)
    return lines

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                correction = 0.2
                steering_measurement = float(batch_sample[3])
                
                # Process centre camera image
                source_path = batch_sample[0]
                centre_image = cv2.imread(data_directory + 'IMG/' + source_path.split('/')[-1])
                centre_image = cv2.cvtColor(centre_image, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
                images.append(centre_image)
                measurements.append(steering_measurement)
                
                # Add flipped images for centre camera
                images.append(cv2.flip(centre_image, 1))
                measurements.append(steering_measurement*-1)
                
                # Process left camera image
                source_path = batch_sample[1]
                left_image = cv2.imread(data_directory + 'IMG/' + source_path.split('/')[-1])
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
                images.append(left_image)
                measurements.append(steering_measurement + correction)
                
                # Process right camera image
                source_path = batch_sample[2]
                right_image = cv2.imread(data_directory + 'IMG/' + source_path.split('/')[-1])
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
                images.append(right_image)
                measurements.append(steering_measurement - correction)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((70,20), (0,0))))
    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.1))
    model.add(Dense(50))
    model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

lines = get_sample_data()
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

batch_size = 32*4
model = build_model()
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator,
                             validation_data=validation_generator,
                             epochs=7,
                             steps_per_epoch=len(train_samples) // batch_size,
                             validation_steps=len(validation_samples) // batch_size)
plot_loss_data(history_object)
model.save('model.h5')
print("model saved")
exit()