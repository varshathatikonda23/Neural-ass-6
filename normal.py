from sklearn.preprocessing import StandardScaler
import numpy as np
sc = StandardScaler()
X_train_normalized = sc.fit_transform()
X_test_normalized = sc.transform()
model_normalized = Sequential()
model_normalized.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model_normalized.add(Dense(64, activation='relu'))
model_normalized.add(Dense(128, activation='relu'))
model_normalized.add(Dense(1, activation='sigmoid'))
model_normalized.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_normalized.fit(X_train_normalized, y_train, epochs=10, batch_size=32, validation_data=(X_test_normalized, y_test))
accuracy_normalized = model_normalized.evaluate(X_test_normalized, y_test)[1]
print("Accuracy with normalization:", accuracy_normalized)
