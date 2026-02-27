import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

if not os.path.exists('model'):
    os.makedirs('model')

np.random.seed(42)
tf.random.set_seed(42)

file_path = 'dataset_augmented.csv' 

try:
    df = pd.read_csv(file_path)

except FileNotFoundError:
    print(f"Error: {file_path} not found.")
    exit()

X = df['cleaned_text'].astype(str)
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vocab_size = 10000
embedding_dim = 100
max_length = 100 
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

print("Creating and saving TOKENIZER...")
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

with open('model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

training_sequences = tokenizer.texts_to_sequences(X_train)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(X_test)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    
    SpatialDropout1D(0.3), 
    
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
    Dropout(0.5),
    
    Bidirectional(LSTM(32, kernel_regularizer=l2(0.001))),
    Dropout(0.5),
    
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True, 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=0.00001,
    verbose=1
)

print("Training LSTM model...")
history = model.fit(
    training_padded, 
    y_train, 
    epochs=10, 
    validation_data=(testing_padded, y_test), 
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model_name = 'model/sexism_model.h5'
model.save(model_name)
print(f"Model saved to {model_name}")

# --- 6. Evaluation & Analysis ---

plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('model/training_history.png')
print("Saved training history plot to model/training_history.png")

# Confusion Matrix
y_pred_prob = model.predict(testing_padded)
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Sexist', 'Sexist'], yticklabels=['Not Sexist', 'Sexist'])
plt.title('LSTM Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('model/confusion_matrix.png')
print("Saved confusion matrix to model/confusion_matrix.png")

report = classification_report(y_test, y_pred, target_names=['Not Sexist', 'Sexist'])
print("\nClassification Report:")
print(report)

with open('model/metrics.txt', 'w') as f:
    f.write("LSTM Model Evaluation Metrics\n")
    f.write("=============================\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
print("Saved metrics to model/metrics.txt")