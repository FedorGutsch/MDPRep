import pandas as pd
import re
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# --- Конфигурация ---
# Имя файла с вашими данными
DATA_FILE = 'normalized.csv'
# Имя файла, куда мы сохраним подготовленный текст для обучения
CORPUS_FILE = 'movie_overviews_corpus.txt'
# Имя файла с предобученными векторами, который вы скачали и распаковали
PRETRAINED_VECTORS_FILE = 'cc.ru.300.vec'
# Имя файла для вашей новой, дообученной модели
FINE_TUNED_MODEL_FILE = 'fine_tuned_movie_model.bin'


# --- Шаг 1: Подготовка корпуса для обучения ---

def fine_tune_model():
    print("Загрузка данных...")
    try:
        real_data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{DATA_FILE}' не найден. Убедитесь, что он находится в той же папке.")
        # Прерываем выполнение, если данных нет
        exit()

    # Функция для очистки и нормализации текста
    def normalize_string(s):
        if not isinstance(s, str):
            return ""
        s = s.lower()
        # Оставляем только русские/английские буквы и пробелы
        s = re.sub(r'[^a-zа-яё\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    print(f"Создание корпуса для обучения в файле '{CORPUS_FILE}'...")
    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        for overview in real_data['overview'].dropna():
            normalized_overview = normalize_string(overview)
            if normalized_overview:
                f.write(normalized_overview + '\n')
    print("Корпус успешно создан.")


    # --- Шаг 2: Дообучение модели FastText ---

    print("Начинаем дообучение модели FastText...")
    print(f"Используем предобученные векторы из: '{PRETRAINED_VECTORS_FILE}'")

    try:
        # Запускаем дообучение. Модель возьмет знания из PRETRAINED_VECTORS_FILE
        # и адаптирует их под ваши данные из CORPUS_FILE.
        model = fasttext.train_unsupervised(
            input=CORPUS_FILE,
            model='skipgram',
            pretrainedVectors=PRETRAINED_VECTORS_FILE,
            dim=300,      # Размерность векторов (должна совпадать с исходной моделью)
            epoch=5,      # Количество эпох обучения. 5 - хорошее начало.
            minCount=3,   # Игнорировать слова, которые встречаются реже 3 раз
            thread=4      # Количество потоков процессора для ускорения
        )

        # Сохраняем нашу новую, дообученную бинарную модель
        model.save_model(FINE_TUNED_MODEL_FILE)
        print(f"\nДообучение завершено! Новая модель сохранена в '{FINE_TUNED_MODEL_FILE}'")

    except ValueError as e:
        if "is not a valid path" in str(e):
            print(f"\nКРИТИЧЕСКАЯ ОШИБКА: Файл '{PRETRAINED_VECTORS_FILE}' не найден!")
            print("Пожалуйста, убедитесь, что вы скачали, распаковали и положили его в ту же папку, что и ноутбук.")
        else:
            print(f"Произошла ошибка: {e}")
        exit()


