import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Функция загрузки аудио файлов и преобразования в мел-спектрограммы
def load_audio_files_and_labels(base_dir, sr=44100, duration=3):
    audio_data = []
    labels = []
    fixed_length = sr * duration  # Фиксированная длина аудио сигнала
    n_fft = 2048
    hop_length = 512
    # Проход по всем подкаталогам (классам)
    for label_dir in os.listdir(base_dir):
        print(label_dir)
        label_path = os.path.join(base_dir, label_dir)
        if os.path.isdir(label_path):
            label = int(label_dir)  # Имя папки — это метка класса
            # Проход по файлам внутри папки
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                if os.path.isfile(file_path):
                    # Загрузка аудио файла
                    y, _ = librosa.load(file_path, sr=sr)
                    # Дополнение или обрезка аудио сигнала до фиксированной длины
                    if len(y) < fixed_length:
                        y = np.pad(y, (0, fixed_length - len(y)), mode='constant')
                    else:
                        y = y[:fixed_length]
                    # Вычисление мел-спектрограммы с фиксированными параметрами
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    audio_data.append(mel_spec_db)
                    labels.append(label)
    return np.array(audio_data), np.array(labels)


# Пример использования
base_dir = "learning_data"  # Папка, содержащая подкаталоги с классами 0, 1, 2 и т.д.
audio_data, labels = load_audio_files_and_labels(base_dir)

# Нормализация данных
audio_data = audio_data / np.max(audio_data)

# Добавление оси канала для сверточной сети
audio_data = np.expand_dims(audio_data, axis=-1)

# Разделение данных на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(audio_data, labels, test_size=0.2, random_state=42)

# Преобразование меток в one-hot encoding
y_train = to_categorical(y_train, num_classes=len(set(labels)))
y_test = to_categorical(y_test, num_classes=len(set(labels)))

# Функция для создания модели
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Параметры модели
input_shape = (audio_data.shape[1], audio_data.shape[2], 1)
num_classes = len(set(labels))

model = create_model(input_shape, num_classes)

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Создание датасетов для обучения и тестирования
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Обучение модели
history = model.fit(train_dataset, epochs=300, validation_data=test_dataset)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')

# График точности
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# График потерь
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Сохранение модели
model.save('audio_classification_model.keras')
