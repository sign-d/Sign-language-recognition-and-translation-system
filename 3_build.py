import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.labels = None
    
    def load_data(self):
        df = pd.read_csv(self.file_path)
        
        self.labels = df.iloc[:, 0].values  
        self.data = df.iloc[:, 1:].values
        
        
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)
        
        return np.array(self.data), np.array(self.labels)


dataset_loader = DatasetLoader("data1.csv")

#load data
data, labels = dataset_loader.load_data()

#split data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
input_dim = train_data.shape[1]

#encode labels
num_classes_labels = len(np.unique(labels))
num_classes_features = data.shape[1]

#one-hot-encorder
train_labels = to_categorical(train_labels, num_classes=num_classes_labels)
test_labels = to_categorical(test_labels, num_classes=num_classes_labels)

print("Data1", num_classes_labels)

#build model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(int(input_dim),)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes_labels, activation='softmax') 
])

print("Data",(train_data.shape[1],))

#compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
# model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
model.fit(train_data, train_labels, epochs=200, batch_size=32, validation_split=0.15)


#evaluate model
results = model.evaluate(test_data, test_labels)
print('Results:', results)
print("Accuracy: ", results[1])
print("Loss: ", results[0])


#save model
model.save('model_1.h5')

