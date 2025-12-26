"""
Скрипт для инициализации базы данных и загрузки фильмов с векторами
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessFunc import normalize_string
from database import Database
import os


def load_e5_model():
    """Загрузить модель E5"""
    print("Загрузка модели E5...")
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    print("Модель E5 загружена!")
    return model


def compute_movie_vectors(df: pd.DataFrame, e5_model: SentenceTransformer, 
                         tfidf_vectorizer: TfidfVectorizer) -> dict:
    """
    Вычислить векторы для всех фильмов
    
    Returns:
        dict: словарь {movie_id: {'e5': ..., 'tfidf': ..., 'combined': ...}}
    """
    print("Вычисление векторов для фильмов...")
    
    movie_vectors = {}
    
    # Подготовка описаний для E5
    print("Подготовка описаний для E5...")
    overviews = []
    genres_list = []
    
    for idx, row in df.iterrows():
        overview = str(row.get('overview', ''))
        genres = str(row.get('genres', ''))
        
        # Нормализация описания
        normalized_overview = normalize_string(overview) if overview else ""
        overviews.append(normalized_overview)
        genres_list.append(genres)
    
    # E5 векторы для описаний
    print("Кодирование описаний через E5 (это может занять время)...")
    passages = ["passage: " + ov for ov in overviews]
    e5_vectors = e5_model.encode(passages, normalize_embeddings=True, 
                                  show_progress_bar=True, batch_size=32)
    
    # TF-IDF векторы для жанров
    print("Вычисление TF-IDF векторов для жанров...")
    tfidf_vectors = tfidf_vectorizer.fit_transform(genres_list).toarray()
    
    # Комбинирование векторов (E5 для описаний + TF-IDF для жанров)
    print("Комбинирование векторов...")
    for idx, row in df.iterrows():
        movie_id = row['movie_id']
        
        # Нормализация векторов
        e5_vec = e5_vectors[idx]
        tfidf_vec = tfidf_vectors[idx]
        
        # Нормализация TF-IDF вектора
        if np.linalg.norm(tfidf_vec) > 0:
            tfidf_vec = tfidf_vec / np.linalg.norm(tfidf_vec)
        
        # Комбинированный вектор (объединение E5 и TF-IDF)
        # Можно масштабировать компоненты или просто объединить
        # Используем простое объединение (можно также взвешивать)
        combined = np.concatenate([e5_vec, tfidf_vec])
        
        movie_vectors[movie_id] = {
            'e5': e5_vec,
            'tfidf': tfidf_vec,
            'combined': combined
        }
    
    print(f"Векторы вычислены для {len(movie_vectors)} фильмов!")
    return movie_vectors


def initialize_database(csv_path: str = "kp_final.csv", db_path: str = "movies_bot.db"):
    """Инициализировать базу данных из CSV файла"""
    
    # Проверка существования CSV файла
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    
    print(f"Загрузка данных из {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} фильмов")
    
    # Инициализация базы данных
    db = Database(db_path)
    
    # Загрузка фильмов в БД
    print("Загрузка фильмов в базу данных...")
    for idx, row in df.iterrows():
        db.add_movie(
            movie_id=int(row['movie_id']),
            movie=str(row['movie']),
            kp_rating=float(row['kp_rating']) if pd.notna(row['kp_rating']) else None,
            movie_duration=int(row['movie_duration']) if pd.notna(row['movie_duration']) else None,
            kp_rating_count=int(row['kp_rating_count']) if pd.notna(row['kp_rating_count']) else None,
            movie_year=int(row['movie_year']) if pd.notna(row['movie_year']) else None,
            genres=str(row['genres']) if pd.notna(row['genres']) else None,
            countries=str(row['countries']) if pd.notna(row['countries']) else None,
            overview=str(row['overview']) if pd.notna(row['overview']) else None,
            poster=str(row['poster']) if pd.notna(row['poster']) else None
        )
    
    print(f"Фильмы загружены в базу данных!")
    
    # Загрузка модели E5
    e5_model = load_e5_model()
    
    # Инициализация TF-IDF векторaйзера для жанров
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        max_features=100,  # Ограничим размерность для жанров
        ngram_range=(1, 2)
    )
    
    # Вычисление векторов
    movie_vectors = compute_movie_vectors(df, e5_model, tfidf_vectorizer)
    
    # Сохранение векторов в БД
    print("Сохранение векторов в базу данных...")
    for movie_id, vectors in movie_vectors.items():
        db.add_movie_vector(
            movie_id=movie_id,
            vector_e5=vectors['e5'],
            vector_tfidf=vectors['tfidf'],
            combined_vector=vectors['combined']
        )
    
    print("База данных инициализирована успешно!")
    print(f"Всего фильмов: {len(movie_vectors)}")
    
    return db, tfidf_vectorizer


if __name__ == "__main__":
    print("Инициализация базы данных фильмов...")
    initialize_database()

