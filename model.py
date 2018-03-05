import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] != 'center':
            lines.append(line)

lines_train, lines_val = train_test_split(lines, test_size=0.2)

def generate_data(lines_input, validation = False):
    while True:
        for line in lines_input:
            images = []
            measurements = []
            path = './data/IMG/'

            # Center cam - normal
            img_center = cv2.imread(path + line[0].split('/')[-1])
            images.append(img_center)
            steering_center = float(line[3])
            measurements.append(steering_center)

            # Do not flip image and do not use left and right cams for validation
            if not validation:
                # Center cam - horizontal flipped
                img_center_flipped = np.fliplr(img_center)
                images.append(img_center_flipped)
                steering_center_flipped = -steering_center
                measurements.append(steering_center_flipped)

                correction = 0.3

                # Left cam - normal
                img_left = cv2.imread(path + line[1].split('/')[-1])
                images.append(img_left)
                steering_left = steering_center + correction
                measurements.append(steering_left)

                # Left cam - horizontal flipped
                img_left_flipped = np.fliplr(img_left)
                images.append(img_left_flipped)
                steering_left_flipped = -steering_left
                measurements.append(steering_left_flipped)

                # Right cam - normal
                img_right = cv2.imread(path + line[2].split('/')[-1])
                images.append(img_right)
                steering_right = steering_center - correction
                measurements.append(steering_right)

                # Right cam - horizontal flipped
                img_right_flipped = np.fliplr(img_right)
                images.append(img_right_flipped)
                steering_right_flipped = -steering_right
                measurements.append(steering_right_flipped)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield (X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

gen_train = generate_data(lines_train, validation = False)
gen_val = generate_data(lines_val, validation = True)

model.compile(loss='mse', optimizer='adam')

epoch = 5

model.fit_generator(gen_train, validation_data = gen_val, nb_val_samples = len(lines_val), samples_per_epoch = len(lines_train) * 6, nb_epoch = epoch)

model.save('model.h5')
