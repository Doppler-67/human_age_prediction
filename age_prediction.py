from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50
import pandas as pd

def load_train(path):

    train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    height_shift_range=0.2, 
    width_shift_range=0.2) 

    labels = pd.read_csv(path + 'labels.csv')
    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'photos/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=67)
    
    return train_gen_flow

def load_test(path):

    test_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255)

    labels = pd.read_csv(path + 'labels.csv')
    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'photos/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=67)
    
    return test_gen_flow



def create_model(input_shape, learning_rate=0.01):

    optimizer = Adam(learning_rate=learning_rate)
    backbone = ResNet50(input_shape=input_shape, include_top=False)
    
    model = Sequential()

    model.add(backbone)
    model.add(GlobalAveragePooling2D())

    model.add(Dense(100, activation='relu'))

    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=optimizer, loss='mean_absolute_error',
                  metrics=['mae'])
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):
  
    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    
    return model


train = load_train(path)
test = load_test(path)
model = create_model(input_shape, 0.0001)
trained_model = train_model(model, train, test)
