import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

def flatten_landmarks(item):
    pose = item['pose'].flatten() if 'pose' in item else np.zeros(132)
    face = item['face'].flatten() if 'face' in item else np.zeros(468 * 3)
    left_hand = item['left_hand'].flatten() if 'left_hand' in item else np.zeros(21 * 3)
    right_hand = item['right_hand'].flatten() if 'right_hand' in item else np.zeros(21 * 3)
    return np.concatenate((pose, face, left_hand, right_hand))

X = [flatten_landmarks(item[0]) for item in data['data']]
X = np.array(X, dtype=np.float32)

y = np.array(data['labels'])

if y.dtype.kind in {'U', 'S'}:
    unique_labels = np.unique(y)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_int[label] for label in y], dtype=np.int32)

y = to_categorical(y)

actions = np.array(unique_labels)
print(actions)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu'), input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, activation='relu')))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(actions.shape[0], activation='softmax'))

optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

kf = KFold(n_splits=5)
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=11, 
              batch_size=32, callbacks=[tb_callback, early_stopping, reduce_lr])

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

print(model.summary())

model.save('model.h5')
model.save_weights('model_weights.h5')
