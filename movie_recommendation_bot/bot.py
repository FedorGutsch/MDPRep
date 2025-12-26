# Основной файл вашего бота (ФИНАЛЬНАЯ ВЕРСИЯ С ИСПРАВЛЕННОЙ КОЛЛАБОРАЦИЕЙ)

import telebot
from telebot import types
import numpy as np
from sentence_transformers import SentenceTransformer
from database import Database
from preprocessFunc import normalize_string
from config import TELEGRAM_TOKEN
from typing import List, Optional, Dict
import random

class RecommendationBot:
    def __init__(self, token: str, db_path: str = "movies_bot.db"):
        self.bot = telebot.TeleBot(token)
        self.db = Database("movies_bot.db")
        print("Загрузка модели E5...")
        self.e5_model = SentenceTransformer('intfloat/multilingual-e5-large')
        print("Модель E5 загружена!")
        self.user_calibration = {}
        self.setup_handlers()

    def setup_handlers(self):
        # ... (код обработчиков без изменений) ...
        @self.bot.message_handler(commands=['start'])
        def start_handler(message): self.handle_start(message)
        @self.bot.message_handler(commands=['restart'])
        def restart_handler(message): self.handle_restart(message)
        @self.bot.message_handler(commands=['search'])
        def search_handler(message): self.handle_search_command(message)
        @self.bot.callback_query_handler(func=lambda call: True)
        def callback_handler(call): self.handle_callback(call)
        @self.bot.message_handler(func=lambda message: True)
        def default_handler(message): self.handle_message(message)

    def handle_start(self, message):
        # ... (код без изменений) ...
        user_id = message.from_user.id; self.db.create_user(user_id)
        user = self.db.get_user(user_id)
        if user and user.get('calibration_complete', 0) == 1:
            self.bot.send_message(user_id, "С возвращением! Ищу для вас что-нибудь интересное...")
            self.send_recommendation(user_id)
        else: self.start_calibration(user_id)
    
    def handle_restart(self, message):
        # ... (код без изменений) ...
        user_id = message.from_user.id
        if user_id in self.user_calibration: del self.user_calibration[user_id]
        self.db.reset_user_calibration(user_id); self.db.create_user(user_id)
        self.bot.reply_to(message, "🔄 Ваш профиль сброшен. Начинаем калибровку заново!")
        self.start_calibration(user_id)

    def start_calibration(self, user_id: int):
        # ... (код без изменений) ...
        self.user_calibration[user_id] = {'ratings': {}, 'shown_movie_ids': set()}
        self.bot.send_message(user_id, 
            "🎬 Давайте настроим ваши рекомендации!\n"
            "Оцените 10 фильмов, которые вы смотрели. Если не смотрели — просто пропустите.")
        self.show_calibration_movie(user_id)

    def show_calibration_movie(self, user_id: int):
        # ... (код без изменений) ...
        if user_id not in self.user_calibration: return
        calibration = self.user_calibration[user_id]
        if len(calibration['ratings']) >= 10: self.complete_calibration(user_id); return
        movie_id = self.get_movie_id_for_calibration(calibration)
        if not movie_id:
            self.bot.send_message(user_id, "❌ Не удалось найти фильмы для калибровки. Попробуйте /restart")
            if user_id in self.user_calibration: del self.user_calibration[user_id]
            return
        calibration['shown_movie_ids'].add(movie_id)
        movie = self.db.get_movie(movie_id)
        if not movie: self.show_calibration_movie(user_id); return
        remaining = 10 - len(calibration['ratings'])
        text = f"Осталось оценить: {remaining}\n\n🎬 <b>{movie['movie']}</b> ({movie.get('movie_year', '')})"
        keyboard = types.InlineKeyboardMarkup(row_width=5)
        buttons = [types.InlineKeyboardButton(f"⭐{i}", callback_data=f"rate_{movie_id}_{i}") for i in range(1, 6)]
        keyboard.add(*buttons)
        keyboard.add(types.InlineKeyboardButton("❌ Не смотрел / Пропустить", callback_data=f"rate_{movie_id}_skip"))
        self.bot.send_message(user_id, text, reply_markup=keyboard, parse_mode='HTML')

    def get_movie_id_for_calibration(self, calibration: dict) -> Optional[int]:
        # ... (код без изменений) ...
        top_movies = self.db.get_top_movies(limit=250, min_rating=0)
        available_movies = [m for m in top_movies if m['movie_id'] not in calibration['shown_movie_ids']]
        return random.choice(available_movies)['movie_id'] if available_movies else None

    def handle_callback(self, call):
        # ... (код без изменений) ...
        user_id = call.from_user.id
        try:
            if call.data.startswith("rate_rec_") or call.data == "get_recommendation":
                if call.data.startswith("rate_rec_"):
                    parts = call.data.split("_"); movie_id, rating = int(parts[2]), int(parts[3])
                    self.update_user_vector_with_rating(user_id, movie_id, rating)
                    self.bot.answer_callback_query(call.id, f"Спасибо! Оценка {rating}⭐ учтена.")
                else: self.bot.answer_callback_query(call.id, "Ищу другой фильм...")
                try: self.bot.delete_message(user_id, call.message.message_id)
                except: pass
                self.send_recommendation(user_id)
            elif call.data.startswith("rate_"):
                if user_id not in self.user_calibration:
                    self.bot.answer_callback_query(call.id, "Калибровка уже завершена."); return
                parts = call.data.split("_"); movie_id, rating_str = int(parts[1]), parts[2]
                if rating_str != "skip":
                    self.user_calibration[user_id]['ratings'][movie_id] = int(rating_str)
                    self.db.add_user_rating(user_id, movie_id, int(rating_str))
                    self.bot.answer_callback_query(call.id, f"Оценка {rating_str}⭐ сохранена!")
                else: self.bot.answer_callback_query(call.id, "Фильм пропущен.")
                self.bot.delete_message(user_id, call.message.message_id)
                self.show_calibration_movie(user_id)
            elif call.data == "start_search":
                self.bot.answer_callback_query(call.id)
                self.bot.send_message(user_id, "🔍 Введите описание фильма для поиска...")
        except Exception as e: print(f"Критическая ошибка в handle_callback: {e}")

    def complete_calibration(self, user_id: int):
        # ... (код без изменений) ...
        if user_id not in self.user_calibration or not self.user_calibration[user_id]['ratings']: return
        ratings = self.user_calibration[user_id]['ratings']
        e5_vectors, tfidf_vectors, combined_vectors, weights = [], [], [], []
        for movie_id, rating in ratings.items():
            movie_vector_data = self.db.get_movie_vector(movie_id)
            if movie_vector_data and movie_vector_data.get('vector_e5') is not None:
                e5_vectors.append(movie_vector_data['vector_e5']); tfidf_vectors.append(movie_vector_data['vector_tfidf'])
                combined_vectors.append(movie_vector_data['combined_vector']); weights.append(rating)
        if not combined_vectors: return
        weights_arr = np.array(weights)
        user_e5 = np.average(np.array(e5_vectors), axis=0, weights=weights_arr)
        user_tfidf = np.average(np.array(tfidf_vectors), axis=0, weights=weights_arr)
        user_combined = np.average(np.array(combined_vectors), axis=0, weights=weights_arr)
        if np.linalg.norm(user_combined) > 0: user_combined /= np.linalg.norm(user_combined)
        if np.linalg.norm(user_e5) > 0: user_e5 /= np.linalg.norm(user_e5)
        if np.linalg.norm(user_tfidf) > 0: user_tfidf /= np.linalg.norm(user_tfidf)
        self.db.save_full_user_profile(user_id, user_e5, user_tfidf, user_combined)
        self.db.set_calibration_complete(user_id, complete=True)
        for movie_id in ratings.keys(): self.db.add_recommendation(user_id, movie_id)
        del self.user_calibration[user_id]
        self.bot.send_message(user_id, "✅ Отлично! Ваш персональный профиль создан. Подбираю первую рекомендацию...")
        self.send_recommendation(user_id)

    def update_user_vector_with_rating(self, user_id: int, movie_id: int, rating: int):
        # ... (код без изменений) ...
        user = self.db.get_user(user_id)
        movie_vector_data = self.db.get_movie_vector(movie_id)
        if not user or not movie_vector_data or user.get('combined_vector') is None: return
        user_vec = user['combined_vector']; movie_vec = movie_vector_data['combined_vector']
        weight, learning_rate = (rating - 3) / 2.0, 0.1
        new_vec = user_vec + learning_rate * weight * (movie_vec - user_vec)
        if np.linalg.norm(new_vec) > 0: new_vec /= np.linalg.norm(new_vec)
        self.db.update_user_combined_vector(user_id, new_vec)
        self.db.increment_ratings_count(user_id)
    
    def find_similar_users(self, user_id: int, user_vector: np.ndarray, top_k: int = 30) -> List[Dict]:
        # ... (код без изменений) ...
        all_users = self.db.get_all_users_with_vectors()
        similarities = []
        for other_user in all_users:
            if other_user['user_id'] == user_id: continue
            if other_user.get('combined_vector') is not None:
                similarity = np.dot(user_vector, other_user['combined_vector'])
                similarities.append({'user_id': other_user['user_id'], 'similarity': similarity})
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ЗДЕСЬ ---
    def generate_recommendation_content(self, user_id: int):
        user = self.db.get_user(user_id)
        if not user or user.get('combined_vector') is None:
            return "❌ Ваш профиль не найден. Пройдите калибровку через /start", None, None
        
        user_vector = user['combined_vector']
        exclude_movie_ids = set(self.db.get_recommended_movie_ids(user_id))
        
        # План А: Коллаборативная фильтрация
        similar_users = self.find_similar_users(user_id, user_vector)
        for sim_user in similar_users:
            highly_rated_movies = self.db.get_highly_rated_movies_by_users(
                [sim_user['user_id']], 
                min_rating=4,  # БЫЛО: 5, СТАЛО: 4. Теперь учитываем и 4⭐, и 5⭐
                exclude_movie_ids=exclude_movie_ids
            )
            if highly_rated_movies:
                recommended_movie_id = highly_rated_movies[0]['movie_id']
                source_text = f"👥 от похожего пользователя (ID: {sim_user['user_id']})"
                self.db.add_recommendation(user_id, recommended_movie_id)
                return self.format_recommendation_message(recommended_movie_id, source_text)

        # План Б: Контентная фильтрация
        all_vectors = self.db.get_all_movie_vectors()
        personal_scores = []
        for movie_data in all_vectors:
            movie_id = movie_data['movie_id']
            if movie_id not in exclude_movie_ids and movie_data.get('combined_vector') is not None:
                similarity = np.dot(user_vector, movie_data['combined_vector'])
                personal_scores.append((similarity, movie_id))
        
        if not personal_scores:
            return "🎉 Поздравляем! Вы видели все фильмы из нашей базы.", None, None
            
        personal_scores.sort(key=lambda x: x[0], reverse=True)
        _, recommended_movie_id = personal_scores[0]
        source_text = "🎯 на основе вашего профиля"
        self.db.add_recommendation(user_id, recommended_movie_id)
        return self.format_recommendation_message(recommended_movie_id, source_text)

    def format_recommendation_message(self, movie_id: int, source_text: str):
        # ... (код без изменений) ...
        movie = self.db.get_movie(movie_id)
        if not movie: return "❌ Ошибка при получении информации о фильме.", None, None
        text = f"🎬 <b>Рекомендация для вас ({source_text}):</b>\n\n<b>{movie['movie']}</b> ({movie.get('movie_year', '')})\n\n"
        if movie.get('overview'): text += f"📝 {movie['overview'][:400]}...\n\n"
        if movie.get('kp_rating'): text += f"⭐ Рейтинг: {movie['kp_rating']}\n"
        keyboard = types.InlineKeyboardMarkup(row_width=5)
        rate_buttons = [types.InlineKeyboardButton(f"⭐{i}", callback_data=f"rate_rec_{movie_id}_{i}") for i in range(1, 6)]
        keyboard.add(*rate_buttons)
        keyboard.add(
            types.InlineKeyboardButton("🎲 Другой фильм", callback_data="get_recommendation"),
            types.InlineKeyboardButton("🔍 Искать", callback_data="start_search")
        )
        return text, keyboard, movie.get('poster')

    # --- Остальные функции без изменений ---
    def send_recommendation(self, user_id: int):
        text, keyboard, poster = self.generate_recommendation_content(user_id)
        if keyboard is None: self.bot.send_message(user_id, text); return
        if poster:
            try: self.bot.send_photo(user_id, poster, caption=text, reply_markup=keyboard, parse_mode='HTML')
            except: self.bot.send_message(user_id, text, reply_markup=keyboard, parse_mode='HTML')
        else: self.bot.send_message(user_id, text, reply_markup=keyboard, parse_mode='HTML')
    # ... (handle_search, handle_message, search_movies, run, __main__) ...
    def handle_search_command(self, message):
        self.bot.send_message(message.from_user.id, "🔍 Введите описание фильма, который хотите найти...")
    def handle_message(self, message):
        user_id, query = message.from_user.id, message.text.strip()
        if not query or len(query) < 3: return
        search_msg = self.bot.reply_to(message, "🔍 Ищу фильмы...")
        try:
            results = self.search_movies_by_description(query, top_k=5)
            if not results:
                self.bot.edit_message_text("❌ Ничего не найдено.", user_id, search_msg.message_id); return
            text = f"🔍 <b>Результаты по запросу «{query}»:</b>\n\n"
            for idx, (movie_id, _) in enumerate(results, 1):
                movie = self.db.get_movie(movie_id)
                if movie: text += f"{idx}. <b>{movie['movie']}</b> ({movie.get('movie_year', '')})\n"
            self.bot.edit_message_text(text, user_id, search_msg.message_id, parse_mode='HTML')
        except Exception as e: print(f"Ошибка при поиске: {e}")
    def search_movies_by_description(self, query: str, top_k: int = 5) -> List[tuple]:
        query_vector = self.e5_model.encode(["query: " + normalize_string(query)], normalize_embeddings=True)[0]
        all_vectors = self.db.get_all_movie_vectors()
        similarities = []
        for movie_data in all_vectors:
            if movie_data.get('vector_e5') is not None:
                similarity = np.dot(query_vector, movie_data['vector_e5'])
                similarities.append((movie_data['movie_id'], similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    def run(self):
        print("Бот запущен...")
        self.bot.polling(none_stop=True)

if __name__ == "__main__":
    if not TELEGRAM_TOKEN: print("ОШИБКА: Телеграм токен не найден.")
    else:
        bot = RecommendationBot(TELEGRAM_TOKEN)
        bot.run()