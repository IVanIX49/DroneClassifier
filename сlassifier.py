import time
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from threading import Thread
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa.display
from tkinter import *
import sys
import os

# Параметры аудиозаписи
sr = 44100  # Частота дискретизации
duration = 3  # Продолжительность записи в секундах


class AudioRecorder:
    def __init__(self, sr=sr, duration=duration):
        self.sr = sr
        self.duration = duration
        self.audio_data = None

    def record_audio(self):
        """Запись аудио в реальном времени."""
        print("Начало записи...")
        self.audio_data = sd.rec(int(self.duration * self.sr), samplerate=self.sr, channels=1, dtype='float32')
        sd.wait()  # Ожидание завершения записи
        print("Запись завершена!")
        return self.audio_data.flatten()  # Возвращаем аудио сигнал как одномерный массив


class AudioClassifier:
    def __init__(self, model_path, class_names):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names

    def classify(self, audio):
        """Классификация аудио с помощью модели."""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=(0, -1))  # Преобразуем в формат для модели

        prediction = self.model.predict(mel_spec_db)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return predicted_class, mel_spec_db

    def get_class_message(self, predicted_class):
        """Возвращает сообщение на основе предсказанного класса."""
        return self.class_names[predicted_class]


class RealTimeAudioClassifierGUI(tk.Tk):
    def __init__(self, recorder, classifier):
        super().__init__()

        self.recorder = recorder
        self.classifier = classifier
        self.stop_threads = False
        self.thread = None

        # Конфигурация окна
        self.title("Система определения класса беспилотника на основе предварительно обученной звуковой модели")
        #self.resizable(False, False)
        self.geometry("1500x800")
        self.configure(bg="#2C2C2C")  # Тёмно-серый фон
        p1 = PhotoImage(file=resource_path('icon.png'))
        self.iconphoto(False, p1)

        # Создание элементов интерфейса
        self.label = tk.Label(self, text="Обнаруженный беспилотник: ", font=("Arial", 24), bg="#2C2C2C", fg="white")
        self.label.pack(pady=10)

        self.canvas_frame = tk.Frame(self, bg="#2C2C2C")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Создание matplotlib Figure для отображения спектрограммы
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Настройка стиля кнопок
        self.create_custom_button()

    def create_custom_button(self):
        # Стиль для кнопок (плавное скругление)
        button_style = {"font": ("Arial", 14), "relief": "flat", "bd": 0, "highlightthickness": 0}

        # Кнопки управления с цветами и скругленными краями
        self.start_button = self.rounded_button(self, "Запустить", "#4CAF50", "#388E3C", command=self.start_classification, **button_style)
        self.start_button.pack(side=tk.LEFT, padx=150, pady=10)

        self.stop_button = self.rounded_button(self, "Остановить", "#FFC107", "#FFA000", command=self.stop_classification, **button_style)
        self.stop_button.pack(side=tk.LEFT, padx=150, pady=10)

        self.exit_button = self.rounded_button(self, "Выход", "#F44336", "#D32F2F", command=self.on_close, **button_style)
        self.exit_button.pack(side=tk.LEFT, padx=150, pady=10)

    def rounded_button(self, parent, text, color, active_color, command=None, **kwargs):
        """Создание скругленной кнопки с заданным цветом и анимацией нажатия"""
        # Использование Canvas для создания кнопки
        canvas = tk.Canvas(parent, width=200, height=50, bg="#2C2C2C", bd=0, highlightthickness=0)
        canvas.pack_propagate(False)

        # Рисуем скругленный прямоугольник
        rect = canvas.create_oval(5, 5, 50, 50, fill=color, outline=color)
        canvas.create_rectangle(25, 5, 175, 50, fill=color, outline=color)
        canvas.create_oval(150, 5, 195, 50, fill=color, outline=color)

        # Добавляем текст в центр кнопки
        button_text = canvas.create_text(100, 27, text=text, fill="white", font=("Arial", 16))

        # Добавляем клик-обработчик для canvas с анимацией
        def on_press(e):
            canvas.itemconfig(rect, fill=active_color)
            command()

        def on_release(e):
            canvas.itemconfig(rect, fill=color)

        canvas.bind("<Button-1>", on_press)
        canvas.bind("<ButtonRelease-1>", on_release)

        return canvas

    def start_classification(self):
        """Запуск классификации аудио в реальном времени."""
        if self.thread is None or not self.thread.is_alive():
            self.stop_threads = False  # Сбрасываем флаг остановки
            self.thread = Thread(target=self.record_and_classify)
            self.thread.start()
            print("Классификация запущена")

    def stop_classification(self):
        """Остановка классификации."""
        if self.thread and self.thread.is_alive():
            self.stop_threads = True  # Устанавливаем флаг остановки
            print("Ожидание завершения потока...")

    def record_and_classify(self):
        """Цикл записи и классификации аудио."""
        while not self.stop_threads:
            # Запись аудио
            try:
                audio = self.recorder.record_audio()
            except:
                self.label.config(text="Ошибка записи! Проверьте исправность устройств!")
                break

            # Классификация аудио
            predicted_class, mel_spec_db = self.classifier.classify(audio)

            # Обновление метки с названием класса
            if predicted_class == 0:
                # set green color
                self.label.config(bg = "green")
            else:
                # set red color
                self.label.config(bg = "red")
            class_message = self.classifier.get_class_message(predicted_class)
            self.update_label(class_message)

            # Обновление спектрограммы
            self.update_spectrogram(mel_spec_db)
 #       exit(0)

    def update_label(self, class_message):
        """Обновление текста метки с предсказанным классом."""
        self.label.config(text=f"Тип беспилотника: {class_message}")

    def update_spectrogram(self, mel_spec_db):
        """Обновление спектрограммы на графике."""
        self.ax.clear()  # Очистка предыдущего графика
        librosa.display.specshow(mel_spec_db[0, :, :, 0], sr=sr, x_axis='time', y_axis='mel', ax=self.ax)
        self.ax.set_title("Спектрограмма записанного звука", color="black")
        self.fig.canvas.draw()

    def on_close(self):
        """Функция, вызываемая при закрытии окна."""
        self.stop_classification()
        if self.thread:
            self.thread.join(timeout=1)  # Ожидание завершения потока с таймаутом
        self.destroy()
        print("Программа закрыта")


# Основной код для работы программы
if __name__ == "__main__":

    def resource_path(relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)


    class_names = ['Не обнаружен', 'Первый тип', 'Второй тип', 'Третий тип', 'Четвертый тип']

    # Создание объектов
    recorder = AudioRecorder(sr=sr, duration=duration)
    classifier = AudioClassifier(model_path=resource_path('audio_classification_model.keras'), class_names=class_names)

    # Создание и запуск графического интерфейса
    app = RealTimeAudioClassifierGUI(recorder, classifier)
    app.protocol("WM_DELETE_WINDOW", app.on_close)  # Обработка закрытия окна
    app.mainloop()
