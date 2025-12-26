# Файл: populate_test_users.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)

import numpy as np
import random
from database import Database

print("--- Скрипт для создания тестовых пользователей ---")

DB_PATH = "movies_bot.db"
NUM_USERS_TO_CREATE = 10
RATINGS_PER_USER = 20

TASTE_PROFILES = {
    "action_fan": ["боевик", "триллер", "фантастика", "приключения"],
    "drama_lover": ["драма", "мелодрама", "биография", "история"],
    "comedy_geek": ["комедия", "семейный", "мультфильм"],
    "sci-fi_nerd": ["фантастика", "фэнтези"],
    "crime_buff": ["криминал", "детектив", "триллер"],
}

def cleanup_old_fake_users(db: Database):
    print("\n[1/4] Очистка старых тестовых пользователей...")
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM users WHERE user_id < 0")
    user_ids_to_delete = [row['user_id'] for row in cursor.fetchall()]
    
    if not user_ids_to_delete:
        print("-> Старые тестовые пользователи не найдены.")
        conn.close()
        return

    placeholders = ','.join('?' * len(user_ids_to_delete))
    cursor.execute(f"DELETE FROM user_ratings WHERE user_id IN ({placeholders})", user_ids_to_delete)
    cursor.execute(f"DELETE FROM recommendations WHERE user_id IN ({placeholders})", user_ids_to_delete)
    cursor.execute(f"DELETE FROM users WHERE user_id IN ({placeholders})", user_ids_to_delete)
    conn.commit()
    conn.close()
    print(f"✅ Удалено {len(user_ids_to_delete)} старых тестовых пользователей.")

def create_fake_users():
    db = Database(DB_PATH)
    cleanup_old_fake_users(db)

    print("\n[2/4] Загрузка данных о фильмах из БД...")
    all_movies = db.get_all_movies_with_genres()
    all_movie_vectors_map = {v['movie_id']: v for v in db.get_all_movie_vectors()}
    
    if not all_movies or not all_movie_vectors_map:
        print("❌ ОШИБКА: В базе данных нет фильмов или векторов. Сначала запустите init_db.py")
        return
    print(f"✅ Загружено {len(all_movies)} фильмов.")

    print(f"\n[3/4] Генерация {NUM_USERS_TO_CREATE} тестовых пользователей...")
    for i in range(NUM_USERS_TO_CREATE):
        user_id = -(i + 1)
        profile_name, keywords = random.choice(list(TASTE_PROFILES.items()))
        
        print(f"\n-> Создание пользователя {user_id} (профиль: {profile_name})...")

        preferred_movies = [m['movie_id'] for m in all_movies if m['genres'] and any(k in m['genres'].lower() for k in keywords)]
        other_movies = [m['movie_id'] for m in all_movies if m['movie_id'] not in preferred_movies]
        
        fake_ratings = {}
        # Высокие оценки для "любимых" жанров
        num_preferred = int(RATINGS_PER_USER * 0.8)
        if len(preferred_movies) > num_preferred:
            for movie_id in random.sample(preferred_movies, num_preferred):
                fake_ratings[movie_id] = random.randint(4, 5)

        # Смешанные оценки для остальных
        num_other = RATINGS_PER_USER - len(fake_ratings)
        if len(other_movies) > num_other:
            for movie_id in random.sample(other_movies, num_other):
                fake_ratings[movie_id] = random.randint(1, 4)

        e5_vectors, tfidf_vectors, combined_vectors, weights = [], [], [], []
        for movie_id, rating in fake_ratings.items():
            movie_vector_data = all_movie_vectors_map.get(movie_id)
            if movie_vector_data and movie_vector_data.get('vector_e5') is not None:
                e5_vectors.append(movie_vector_data['vector_e5'])
                tfidf_vectors.append(movie_vector_data['vector_tfidf'])
                combined_vectors.append(movie_vector_data['combined_vector'])
                weights.append(rating)
        
        if not combined_vectors: continue

        weights_arr = np.array(weights)
        user_e5 = np.average(np.array(e5_vectors), axis=0, weights=weights_arr)
        user_tfidf = np.average(np.array(tfidf_vectors), axis=0, weights=weights_arr)
        user_combined = np.average(np.array(combined_vectors), axis=0, weights=weights_arr)
        
        if np.linalg.norm(user_combined) > 0: user_combined /= np.linalg.norm(user_combined)
        if np.linalg.norm(user_e5) > 0: user_e5 /= np.linalg.norm(user_e5)
        if np.linalg.norm(user_tfidf) > 0: user_tfidf /= np.linalg.norm(user_tfidf)

        db.create_user(user_id)
        db.save_full_user_profile(user_id, user_e5, user_tfidf, user_combined)
        db.set_calibration_complete(user_id, complete=True)
        for movie_id, rating in fake_ratings.items(): db.add_user_rating(user_id, movie_id, rating)
        
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET ratings_count = ? WHERE user_id = ?", (len(fake_ratings), user_id))
        conn.commit()
        conn.close()

    print("\n[4/4] Финальная проверка...")
    conn = db.get_connection()
    cursor = conn.cursor()
    count = cursor.execute("SELECT COUNT(*) FROM users WHERE user_id < 0 AND calibration_complete = 1").fetchone()[0]
    conn.close()
    print(f"✅ В базе данных теперь {count} готовых к работе тестовых пользователей.")

if __name__ == "__main__":
    create_fake_users()
    print("\n--- Работа скрипта завершена ---")