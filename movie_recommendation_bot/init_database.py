# Файл: init_db.py (Надежная версия)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessFunc import normalize_string
from database import Database
import os

# --- Конфигурация ---
# Скрипт будет искать CSV файл в ЭТОЙ ЖЕ папке
CSV_FILE_NAME = "kinopoisk-top250.csv"
DB_FILE_NAME = "movies_bot.db"

def initialize_database():
    """
    Основная функция для создания и наполнения базы данных.
    """
    print("--- НАЧАЛО ИНИЦИАЛИЗАЦИИ БАЗЫ ДАННЫХ ---")

    # --- ШАГ 1: Проверка и чтение CSV ---
    print(f"\n[1/5] Поиск CSV файла '{CSV_FILE_NAME}' в текущей папке...")
    if not os.path.exists(CSV_FILE_NAME):
        print(f"❌ ОШИБКА: Файл '{CSV_FILE_NAME}' не найден!")
        print(f"-> Убедитесь, что он лежит в той же папке, что и этот скрипт.")
        return

    print("✅ Файл найден. Загрузка данных...")
    df = pd.read_csv(CSV_FILE_NAME)
    print(f"✅ Загружено {len(df)} строк из CSV.")

    # --- ШАГ 2: Создание БД ---
    print(f"\n[2/5] Создание файла базы данных '{DB_FILE_NAME}'...")
    if os.path.exists(DB_FILE_NAME):
        os.remove(DB_FILE_NAME)
        print("-> Старый файл базы данных удален для чистого старта.")
    db = Database(db_path=DB_FILE_NAME)
    
    loaded_count = 0
    for idx, row in df.iterrows():
        try:
            db.add_movie(
                movie_id=idx + 1,
                movie=str(row.get('movie', 'Без названия')),
                kp_rating=float(row.get('rating_ball', 0)),
                movie_year=int(row.get('year', 0)),
                countries=str(row.get('country', '')),
                overview=str(row.get('overview', '')),
                poster=str(row.get('url_logo', '')),
                genres=""  # В kinopoisk-top250.csv нет жанров
            )
            loaded_count += 1
        except Exception as e:
            print(f"-> Предупреждение: не удалось загрузить строку {idx}. Ошибка: {e}")
    print(f"✅ В таблицу 'movies' загружено {loaded_count} фильмов.")

    # --- ШАГ 3: Загрузка AI модели ---
    print("\n[3/5] Загрузка AI модели E5 (это может занять время)...")
    e5_model = SentenceTransformer('intfloat/multilingual-e5-large')
    print("✅ Модель E5 загружена.")

    # --- ШАГ 4: Вычисление векторов ---
    print("\n[4/5] Вычисление векторов для всех фильмов...")
    passages = ["passage: " + normalize_string(str(ov)) for ov in df['overview'].fillna('')]
    e5_vectors = e5_model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
    
    # Создаем "пустые" векторы для жанров, т.к. их нет в исходном файле
    tfidf_vectors = np.zeros((len(df), 100))
    
    print("✅ Векторы рассчитаны. Сохранение в базу данных...")
    for idx in range(len(df)):
        db.add_movie_vector(
            movie_id=idx + 1,
            vector_e5=e5_vectors[idx],
            vector_tfidf=tfidf_vectors[idx],
            combined_vector=np.concatenate([e5_vectors[idx], tfidf_vectors[idx]])
        )
    print(f"✅ Векторы для {len(df)} фильмов сохранены.")

    # --- ШАГ 5: Финальная проверка ---
    print("\n[5/5] Финальная проверка целостности базы данных...")
    conn = db.get_connection()
    movies_count = conn.cursor().execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    vectors_count = conn.cursor().execute("SELECT COUNT(*) FROM movie_vectors").fetchone()[0]
    conn.close()
    
    if movies_count == vectors_count and movies_count > 0:
        print("\n🎉🎉🎉 БАЗА ДАННЫХ УСПЕШНО СОЗДАНА! 🎉🎉🎉")
        print(f"-> Создан файл '{DB_FILE_NAME}' с {movies_count} фильмами.")
    else:
        print("\n❌ ОШИБКА: Финальная проверка не пройдена. Что-то пошло не так.")

if __name__ == "__main__":
    initialize_database()