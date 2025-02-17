### README.md
# Система определения класса беспилотника на основе предварительно обученной звуковой модели

## Описание

**Real-Time Audio Classifier** — это инструмент для классификации звуковых сигналов, используемых беспилотными летательными аппаратами (БЛА). Программа предназначена для реального времени анализа аудио потока и определения типа БЛА на основе предварительно обученной модели машинного обучения.

## Функциональные возможности

- **Запись аудио:** Возможность записи звукового сигнала в реальном времени с заданной частотой дискретизации и продолжительностью.
- **Классификация:** Анализ записанного аудио с использованием предварительно обученной нейронной сети для определения типа БЛА.
- **Графический интерфейс:** Интуитивно понятный интерфейс для управления процессом записи и классификации, а также для отображения результатов анализа.
- **Отображение спектрограммы:** Визуализация спектрограммы записанного звука для анализа его характеристик.

## Установка и запуск

1. **Установите необходимые библиотеки:**

   ```bash
   pip install sounddevice numpy librosa tensorflow matplotlib
   ```
## Использование

- **Запустить запись и классификацию:** Нажмите кнопку "Запустить". Программа начнет записывать аудио и сразу же будет классифицировать его.
- **Остановить процесс:** Нажмите кнопку "Остановить". Процесс записи и классификации будет остановлен.
- **Выход из программы:** Нажмите кнопку "Выход" или закройте окно программы через стандартную системную панель управления окном.

## Архитектура программы

1. **AudioRecorder:** Класс для записи аудио в реальном времени с заданной частотой дискретизации и продолжительностью.
2. **AudioClassifier:** Класс для классификации аудио с использованием предварительно обученной модели TensorFlow. Выполняет преобразование аудиосигнала в мел-спектрограмму и передает ее в модель для предсказания.
3. **RealTimeAudioClassifierGUI:** Графический интерфейс пользователя, основанный на Tkinter, который предоставляет функционал для управления процессом записи и классификации, а также для отображения результатов анализа.

## Лицензия

Этот проект распространяется под лицензией MIT. Полный текст лицензии доступен в файле [LICENSE](LICENSE).

## Авторы

- [IVanIX])

## Поддержка

Если у вас возникнут вопросы или проблемы с программой, пожалуйста, создайте issue в данном репозитории. Мы будем рады помочь вам!

## Дополнительная информация

- [Документация по библиотекам](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html)
- [Примеры использования TensorFlow](https://www.tensorflow.org/tutorials)
- [Ресурсы по работе с звуком в Python](https://python-sounddevice.readthedocs.io/en/0.4.1/)

---

*Дата последнего обновления: [17.02.2025]*

---

**Спасибо за использование нашей системы определения класса беспилотников! Мы стремимся сделать процесс анализа звуковых сигналов простым и эффективным.**
