"""
Модуль для работы с базой данных SQLite3
"""
import sqlite3
import numpy as np
import json
from typing import Optional, List, Tuple


class Database:
    def __init__(self, db_path: str = "movies_bot.db"):
        """Инициализация подключения к базе данных"""
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Получить соединение с базой данных"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Создание таблиц в базе данных"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Таблица пользователей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                vector_e5 TEXT,
                vector_tfidf TEXT,
                combined_vector TEXT,
                calibration_complete INTEGER DEFAULT 0,
                ratings_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица фильмов (оригинальные данные)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                movie_id INTEGER PRIMARY KEY,
                movie TEXT NOT NULL,
                kp_rating REAL,
                movie_duration INTEGER,
                kp_rating_count INTEGER,
                movie_year INTEGER,
                genres TEXT,
                countries TEXT,
                overview TEXT,
                poster TEXT
            )
        """)
        
        # Таблица векторов фильмов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movie_vectors (
                movie_id INTEGER PRIMARY KEY,
                vector_e5 TEXT,
                vector_tfidf TEXT,
                combined_vector TEXT,
                FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"База данных инициализирована: {self.db_path}")
    
    def add_movie(self, movie_id: int, movie: str, kp_rating: float = None,
                  movie_duration: int = None, kp_rating_count: int = None,
                  movie_year: int = None, genres: str = None,
                  countries: str = None, overview: str = None, poster: str = None):
        """Добавить фильм в базу данных"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO movies 
            (movie_id, movie, kp_rating, movie_duration, kp_rating_count, 
             movie_year, genres, countries, overview, poster)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (movie_id, movie, kp_rating, movie_duration, kp_rating_count,
              movie_year, genres, countries, overview, poster))
        
        conn.commit()
        conn.close()
    
    def add_movie_vector(self, movie_id: int, vector_e5: np.ndarray = None,
                        vector_tfidf: np.ndarray = None, combined_vector: np.ndarray = None):
        """Добавить вектор фильма в базу данных"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Конвертация numpy массивов в JSON строки
        vec_e5_str = json.dumps(vector_e5.tolist()) if vector_e5 is not None else None
        vec_tfidf_str = json.dumps(vector_tfidf.tolist()) if vector_tfidf is not None else None
        vec_combined_str = json.dumps(combined_vector.tolist()) if combined_vector is not None else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO movie_vectors 
            (movie_id, vector_e5, vector_tfidf, combined_vector)
            VALUES (?, ?, ?, ?)
        """, (movie_id, vec_e5_str, vec_tfidf_str, vec_combined_str))
        
        conn.commit()
        conn.close()
    
    def get_user(self, user_id: int) -> Optional[dict]:
        """Получить данные пользователя"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        user = dict(row)
        # Конвертация JSON строк обратно в numpy массивы
        if user.get('vector_e5'):
            user['vector_e5'] = np.array(json.loads(user['vector_e5']))
        if user.get('vector_tfidf'):
            user['vector_tfidf'] = np.array(json.loads(user['vector_tfidf']))
        if user.get('combined_vector'):
            user['combined_vector'] = np.array(json.loads(user['combined_vector']))
        
        return user
    
    def create_user(self, user_id: int):
        """Создать нового пользователя"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO users (user_id)
            VALUES (?)
        """, (user_id,))
        
        conn.commit()
        conn.close()
    
    def update_user_vector(self, user_id: int, vector_e5: np.ndarray = None,
                          vector_tfidf: np.ndarray = None, combined_vector: np.ndarray = None):
        """Обновить вектор пользователя"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        updates = []
        values = []
        
        if vector_e5 is not None:
            updates.append("vector_e5 = ?")
            values.append(json.dumps(vector_e5.tolist()))
        
        if vector_tfidf is not None:
            updates.append("vector_tfidf = ?")
            values.append(json.dumps(vector_tfidf.tolist()))
        
        if combined_vector is not None:
            updates.append("combined_vector = ?")
            values.append(json.dumps(combined_vector.tolist()))
        
        if updates:
            values.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?"
            cursor.execute(query, values)
            conn.commit()
        
        conn.close()
    
    def set_calibration_complete(self, user_id: int, complete: bool = True):
        """Установить флаг завершения калибровки"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET calibration_complete = ?, ratings_count = ratings_count + 1
            WHERE user_id = ?
        """, (1 if complete else 0, user_id))
        
        conn.commit()
        conn.close()
    
    def increment_ratings_count(self, user_id: int):
        """Увеличить счетчик оценок пользователя"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET ratings_count = ratings_count + 1
            WHERE user_id = ?
        """, (user_id,))
        
        conn.commit()
        conn.close()
    
    def get_movie(self, movie_id: int) -> Optional[dict]:
        """Получить данные фильма"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM movies WHERE movie_id = ?", (movie_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return dict(row)
    
    def get_movie_vector(self, movie_id: int) -> Optional[dict]:
        """Получить вектор фильма"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM movie_vectors WHERE movie_id = ?", (movie_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        vector_data = dict(row)
        # Конвертация JSON строк обратно в numpy массивы
        if vector_data.get('vector_e5'):
            vector_data['vector_e5'] = np.array(json.loads(vector_data['vector_e5']))
        if vector_data.get('vector_tfidf'):
            vector_data['vector_tfidf'] = np.array(json.loads(vector_data['vector_tfidf']))
        if vector_data.get('combined_vector'):
            vector_data['combined_vector'] = np.array(json.loads(vector_data['combined_vector']))
        
        return vector_data
    
    def get_top_movies(self, limit: int = 250, min_rating: float = 8.0) -> List[dict]:
        """Получить топ фильмов по рейтингу"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT movie_id FROM movies 
            WHERE kp_rating >= ?
            ORDER BY kp_rating DESC
            LIMIT ?
        """, (min_rating, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_all_movie_ids(self) -> List[int]:
        """Получить все ID фильмов"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT movie_id FROM movies")
        rows = cursor.fetchall()
        conn.close()
        
        return [row['movie_id'] for row in rows]
    
    def movie_exists(self, movie_id: int) -> bool:
        """Проверить существование фильма"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM movies WHERE movie_id = ?", (movie_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        
        return exists

