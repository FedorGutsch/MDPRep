# Файл: database.py (ФИНАЛЬНАЯ ВЕРСИЯ)

import sqlite3
import numpy as np
import json
from typing import Optional, List, Dict, Set

class Database:
    def __init__(self, db_path: str = "movies_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        # ... (код создания таблиц) ...
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY, vector_e5 TEXT, vector_tfidf TEXT,
                combined_vector TEXT, calibration_complete INTEGER DEFAULT 0,
                ratings_count INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                movie_id INTEGER PRIMARY KEY, movie TEXT NOT NULL, kp_rating REAL,
                movie_duration INTEGER, kp_rating_count INTEGER, movie_year INTEGER,
                genres TEXT, countries TEXT, overview TEXT, poster TEXT
            )""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movie_vectors (
                movie_id INTEGER PRIMARY KEY, vector_e5 TEXT, vector_tfidf TEXT,
                combined_vector TEXT, FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
            )""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL, recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, movie_id)
            )""")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL, rating INTEGER NOT NULL,
                rated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_id, movie_id)
            )""")
        conn.commit()
        conn.close()

    def save_full_user_profile(self, user_id: int, vec_e5: np.ndarray, vec_tfidf: np.ndarray, vec_combined: np.ndarray):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET vector_e5 = ?, vector_tfidf = ?, combined_vector = ? WHERE user_id = ?",
            (json.dumps(vec_e5.tolist()), json.dumps(vec_tfidf.tolist()), json.dumps(vec_combined.tolist()), user_id)
        )
        conn.commit()
        conn.close()

    def update_user_combined_vector(self, user_id: int, combined_vector: np.ndarray):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET combined_vector = ? WHERE user_id = ?",
            (json.dumps(combined_vector.tolist()), user_id)
        )
        conn.commit()
        conn.close()

    def add_movie(self, **kwargs):
        conn = self.get_connection()
        cursor = conn.cursor()
        cols, placeholders = ', '.join(kwargs.keys()), ', '.join('?' * len(kwargs))
        cursor.execute(f"INSERT OR REPLACE INTO movies ({cols}) VALUES ({placeholders})", tuple(kwargs.values()))
        conn.commit()
        conn.close()

    def add_movie_vector(self, movie_id: int, vector_e5: np.ndarray, vector_tfidf: np.ndarray, combined_vector: np.ndarray):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO movie_vectors (movie_id, vector_e5, vector_tfidf, combined_vector) VALUES (?, ?, ?, ?)",
            (movie_id, json.dumps(vector_e5.tolist()), json.dumps(vector_tfidf.tolist()), json.dumps(combined_vector.tolist()))
        )
        conn.commit()
        conn.close()

    def get_user(self, user_id: int) -> Optional[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if not row: return None
        user = dict(row)
        if user.get('combined_vector'):
            user['combined_vector'] = np.array(json.loads(user['combined_vector']))
        return user

    def create_user(self, user_id: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        conn.commit()
        conn.close()

    def set_calibration_complete(self, user_id: int, complete: bool = True):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET calibration_complete = ? WHERE user_id = ?", (1 if complete else 0, user_id))
        conn.commit()
        conn.close()

    def increment_ratings_count(self, user_id: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET ratings_count = ratings_count + 1 WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()

    def reset_user_calibration(self, user_id: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET calibration_complete = 0, vector_e5 = NULL, vector_tfidf = NULL, combined_vector = NULL, ratings_count = 0 WHERE user_id = ?",
            (user_id,)
        )
        cursor.execute("DELETE FROM recommendations WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM user_ratings WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()

    def get_movie(self, movie_id: int) -> Optional[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE movie_id = ?", (movie_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_movie_vector(self, movie_id: int) -> Optional[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM movie_vectors WHERE movie_id = ?", (movie_id,))
        row = cursor.fetchone()
        conn.close()
        if not row: return None
        vector_data = dict(row)
        for key in ['vector_e5', 'vector_tfidf', 'combined_vector']:
            if vector_data.get(key): vector_data[key] = np.array(json.loads(vector_data[key]))
        return vector_data

    def get_all_movie_vectors(self) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM movie_vectors")
        rows = cursor.fetchall()
        conn.close()
        all_vectors = []
        for row in rows:
            vector_data = dict(row)
            try:
                for key in ['vector_e5', 'vector_tfidf', 'combined_vector']:
                    if vector_data.get(key): vector_data[key] = np.array(json.loads(vector_data[key]))
                all_vectors.append(vector_data)
            except Exception as e:
                print(f"Ошибка при обработке векторов для movie_id {vector_data.get('movie_id')}: {e}")
        return all_vectors

    def get_top_movies(self, limit: int = 250, min_rating: float = 0.0) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT movie_id FROM movies WHERE kp_rating >= ? ORDER BY kp_rating DESC LIMIT ?", (min_rating, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def add_recommendation(self, user_id: int, movie_id: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO recommendations (user_id, movie_id) VALUES (?, ?)", (user_id, movie_id))
        conn.commit()
        conn.close()

    def get_recommended_movie_ids(self, user_id: int) -> List[int]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT movie_id FROM recommendations WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [row['movie_id'] for row in rows]

    def add_user_rating(self, user_id: int, movie_id: int, rating: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO user_ratings (user_id, movie_id, rating) VALUES (?, ?, ?)", (user_id, movie_id, rating))
        conn.commit()
        conn.close()
    
    def get_all_movies_with_genres(self) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT movie_id, genres FROM movies")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # --- ВОТ НОВЫЕ МЕТОДЫ ---

    def get_all_users_with_vectors(self) -> List[Dict]:
        """Получить всех пользователей с завершенной калибровкой и их векторами."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, combined_vector 
            FROM users 
            WHERE calibration_complete = 1 AND combined_vector IS NOT NULL
        """)
        rows = cursor.fetchall()
        conn.close()
        
        users = []
        for row in rows:
            if row['combined_vector']:
                try:
                    users.append({
                        'user_id': row['user_id'],
                        'combined_vector': np.array(json.loads(row['combined_vector']))
                    })
                except (json.JSONDecodeError, TypeError):
                    continue
        return users

    def get_highly_rated_movies_by_users(self, user_ids: List[int], min_rating: int = 4, exclude_movie_ids: Set[int] = None) -> List[Dict]:
        """Получить фильмы, которые высоко оценили указанные пользователи."""
        if not user_ids:
            return []
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(user_ids))
        query = f"""
            SELECT 
                movie_id, AVG(rating) as avg_rating, COUNT(*) as rating_count
            FROM user_ratings
            WHERE user_id IN ({placeholders}) AND rating >= ?
        """
        params = user_ids + [min_rating]
        
        if exclude_movie_ids:
            exclude_placeholders = ','.join('?' * len(exclude_movie_ids))
            query += f" AND movie_id NOT IN ({exclude_placeholders})"
            params.extend(list(exclude_movie_ids))
        
        query += " GROUP BY movie_id ORDER BY rating_count DESC, avg_rating DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]