"""
Телеграм-бот для рекомендации фильмов на основе персонального профиля пользователя
"""
import telebot
from telebot import types
import numpy as np
from sentence_transformers import SentenceTransformer
from database import Database
from preprocessFunc import normalize_string
from config import TELEGRAM_TOKEN  # Убедитесь, что у вас есть этот файл с токеном
from typing import List, Optional
import random

class RecommendationBot:
    def __init__(self, token: str, db_path: str = "movies_bot.db"):
        """Инициализация бота"""
        self.bot = telebot.TeleBot(token)
        self.db = Database(db_path)
        
        print("Загрузка модели E5...")
        self.e5_model = SentenceTransformer('intfloat/multilingual-e5-small')
        print("Модель E5 загружена!")
        
        self.user_calibration = {}
        self.setup_handlers()
    
    def setup_handlers(self):
        """Настройка обработчиков сообщений"""
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
        """Обработка команды /start"""
        user_id = message.from_user.id
        self.db.create_user(user_id)
        user = self.db.get_user(user_id)
        
        if user and user.get('calibration_complete', 0) == 1:
            self.bot.send_message(user_id, "С возвращением! Ищу для вас что-нибудь интересное...")
            self.send_recommendation(user_id)
        else:
            self.start_calibration(user_id)
    
    def handle_restart(self, message):
        """Обработка команды /restart"""
        user_id = message.from_user.id
        if user_id in self.user_calibration:
            del self.user_calibration[user_id]
        
        self.db.reset_user_calibration(user_id)
        self.db.create_user(user_id)
        
        self.bot.reply_to(message, "🔄 Ваш профиль сброшен. Начинаем калибровку заново!")
        self.start_calibration(user_id)
    
    def start_calibration(self, user_id: int):
        """Начать процесс калибровки"""
        self.user_calibration[user_id] = {'ratings': {}, 'shown_movie_ids': set()}
        self.bot.send_message(user_id, 
            "🎬 Давайте настроим ваши рекомендации!\n"
            "Оцените 10 фильмов, которые вы смотрели. Если не смотрели — просто пропустите.")
        self.show_calibration_movie(user_id)
    
    def show_calibration_movie(self, user_id: int):
        """Показать фильм для калибровки"""
        if user_id not in self.user_calibration: return
        calibration = self.user_calibration[user_id]
        
        if len(calibration['ratings']) >= 10:
            self.complete_calibration(user_id)
            return
        
        movie_id = self.get_movie_id_for_calibration(calibration)
        
        if not movie_id:
            self.bot.send_message(user_id, "❌ Не удалось найти фильмы для калибровки. Попробуйте /restart")
            if user_id in self.user_calibration: del self.user_calibration[user_id]
            return
            
        calibration['shown_movie_ids'].add(movie_id)
        movie = self.db.get_movie(movie_id)
        if not movie:
            self.show_calibration_movie(user_id); return
        
        remaining = 10 - len(calibration['ratings'])
        text = f"Осталось оценить: {remaining}\n\n🎬 <b>{movie['movie']}</b> ({movie.get('movie_year', '')})"
        
        keyboard = types.InlineKeyboardMarkup(row_width=5)
        buttons = [types.InlineKeyboardButton(f"⭐{i}", callback_data=f"rate_{movie_id}_{i}") for i in range(1, 6)]
        keyboard.add(*buttons)
        keyboard.add(types.InlineKeyboardButton("❌ Не смотрел / Пропустить", callback_data=f"rate_{movie_id}_skip"))
        
        self.bot.send_message(user_id, text, reply_markup=keyboard, parse_mode='HTML')

    def get_movie_id_for_calibration(self, calibration: dict) -> Optional[int]:
        top_movies = self.db.get_top_movies(limit=250, min_rating=7.5)
        available_movies = [m for m in top_movies if m['movie_id'] not in calibration['shown_movie_ids']]
        return random.choice(available_movies)['movie_id'] if available_movies else None

    # --- КЛЮЧЕВАЯ ИСПРАВЛЕННАЯ ФУНКЦИЯ ---
    def handle_callback(self, call):
        """Обработка всех нажатий на кнопки"""
        user_id = call.from_user.id
        
        try:
            # --- Логика для кнопок под РЕКОМЕНДАЦИЕЙ ---
            if call.data.startswith("rate_rec_") or call.data == "get_recommendation":
                print(f"Пользователь {user_id} нажал кнопку рекомендации: {call.data}")

                # Шаг 1: Выполняем действие
                if call.data.startswith("rate_rec_"):
                    parts = call.data.split("_")
                    movie_id, rating = int(parts[2]), int(parts[3])
                    self.update_user_vector_with_rating(user_id, movie_id, rating)
                    self.bot.answer_callback_query(call.id, f"Спасибо! Оценка {rating}⭐ учтена.")
                else:
                    self.bot.answer_callback_query(call.id, "Ищу другой фильм...")
                
                # Шаг 2: Удаляем старое сообщение
                try:
                    self.bot.delete_message(user_id, call.message.message_id)
                    print(f"Сообщение {call.message.message_id} удалено для {user_id}.")
                except Exception as e:
                    print(f"Не удалось удалить сообщение {call.message.message_id}: {e}")

                # Шаг 3: Отправляем новое
                self.send_recommendation(user_id)

            # --- Логика для кнопок КАЛИБРОВКИ ---
            elif call.data.startswith("rate_"):
                print(f"Пользователь {user_id} нажал кнопку калибровки: {call.data}")
                
                # Шаг 1: Выполняем действие
                if user_id not in self.user_calibration:
                    self.bot.answer_callback_query(call.id, "Калибровка уже завершена. Используйте /restart, чтобы начать заново.")
                    return

                calibration = self.user_calibration[user_id]
                parts = call.data.split("_")
                movie_id, rating_str = int(parts[1]), parts[2]

                if rating_str != "skip":
                    rating = int(rating_str)
                    calibration['ratings'][movie_id] = rating
                    self.db.add_user_rating(user_id, movie_id, rating)
                    self.bot.answer_callback_query(call.id, f"Оценка {rating}⭐ сохранена!")
                else:
                    self.bot.answer_callback_query(call.id, "Фильм пропущен.")

                # Шаг 2: Удаляем старое сообщение
                self.bot.delete_message(user_id, call.message.message_id)
                print(f"Сообщение калибровки {call.message.message_id} удалено для {user_id}.")

                # Шаг 3: Показываем следующий шаг калибровки
                self.show_calibration_movie(user_id)

            # --- Логика для кнопки ПОИСКА ---
            elif call.data == "start_search":
                self.bot.answer_callback_query(call.id)
                self.bot.send_message(user_id, "🔍 Введите описание фильма для поиска...")

        except Exception as e:
            print(f"Критическая ошибка в handle_callback для user {user_id}: {e}")
            self.bot.answer_callback_query(call.id, "Произошла непредвиденная ошибка.")

    def complete_calibration(self, user_id: int):
        """Завершение калибровки"""
        if user_id not in self.user_calibration or not self.user_calibration[user_id]['ratings']:
            self.bot.send_message(user_id, "❌ Вы не оценили достаточно фильмов. Начните заново /start")
            if user_id in self.user_calibration: del self.user_calibration[user_id]
            return
        
        ratings = self.user_calibration[user_id]['ratings']
        vectors, weights = [], []
        
        for movie_id, rating in ratings.items():
            movie_vector_data = self.db.get_movie_vector(movie_id)
            if movie_vector_data and movie_vector_data.get('combined_vector') is not None:
                vectors.append(movie_vector_data['combined_vector'])
                weights.append(rating)
        
        if not vectors:
            self.bot.send_message(user_id, "❌ Ошибка при вычислении профиля. Попробуйте /restart.")
            del self.user_calibration[user_id]; return

        user_vector = np.average(np.array(vectors), axis=0, weights=np.array(weights))
        if np.linalg.norm(user_vector) > 0: user_vector /= np.linalg.norm(user_vector)
        
        self.db.update_user_vector(user_id=user_id, combined_vector=user_vector)
        self.db.set_calibration_complete(user_id, complete=True)
        for movie_id in ratings.keys(): self.db.add_recommendation(user_id, movie_id)
        
        del self.user_calibration[user_id]
        
        self.bot.send_message(user_id, "✅ Отлично! Ваш персональный профиль создан. Подбираю первую рекомендацию...")
        self.send_recommendation(user_id)

    def update_user_vector_with_rating(self, user_id: int, movie_id: int, rating: int):
        """Обновление вектора пользователя"""
        user = self.db.get_user(user_id)
        movie_vector_data = self.db.get_movie_vector(movie_id)
        if not user or not movie_vector_data or user.get('combined_vector') is None: return

        user_vec = user['combined_vector']
        movie_vec = movie_vector_data['combined_vector']
        
        weight = (rating - 3) / 2.0
        learning_rate = 0.1
        
        new_vec = user_vec + learning_rate * weight * (movie_vec - user_vec)
        if np.linalg.norm(new_vec) > 0: new_vec /= np.linalg.norm(new_vec)

        self.db.update_user_vector(user_id=user_id, combined_vector=new_vec)
        self.db.add_recommendation(user_id, movie_id)
        self.db.increment_ratings_count(user_id)

    def generate_recommendation_content(self, user_id: int):
        """Подбор рекомендации"""
        user = self.db.get_user(user_id)
        if not user or user.get('combined_vector') is None:
            return "❌ Ваш профиль не найден. Пройдите калибровку через /start", None, None
        
        user_vector = user['combined_vector']
        exclude_movie_ids = set(self.db.get_recommended_movie_ids(user_id))
        
        all_vectors = self.db.get_all_movie_vectors()
        scores = []
        for movie_data in all_vectors:
            movie_id = movie_data['movie_id']
            if movie_id in exclude_movie_ids: continue
            
            movie_vector = movie_data.get('combined_vector')
            if movie_vector is not None:
                similarity = np.dot(user_vector, movie_vector)
                scores.append((similarity, movie_id))
        
        if not scores:
            return "🎉 Похоже, вы уже видели все фильмы! Используйте /restart, чтобы сбросить историю.", None, None
        
        scores.sort(key=lambda x: x[0], reverse=True)
        top_scores = scores[:5]
        if not top_scores:
             return "🎉 Не могу найти подходящих рекомендаций. Попробуйте /restart.", None, None

        _, recommended_movie_id = random.choice(top_scores)
        
        movie = self.db.get_movie(recommended_movie_id)
        if not movie: return "❌ Ошибка при получении информации о фильме.", None, None
        
        text = f"🎬 <b>Рекомендация для вас:</b>\n\n<b>{movie['movie']}</b> ({movie.get('movie_year', '')})\n\n"
        if movie.get('overview'): text += f"📝 {movie['overview'][:400]}...\n\n"
        if movie.get('kp_rating'): text += f"⭐ Рейтинг: {movie['kp_rating']}\n"
        
        keyboard = types.InlineKeyboardMarkup(row_width=5)
        rate_buttons = [types.InlineKeyboardButton(f"⭐{i}", callback_data=f"rate_rec_{recommended_movie_id}_{i}") for i in range(1, 6)]
        keyboard.add(*rate_buttons)
        keyboard.add(
            types.InlineKeyboardButton("🎲 Другой фильм", callback_data="get_recommendation"),
            types.InlineKeyboardButton("🔍 Искать", callback_data="start_search")
        )
        return text, keyboard, movie.get('poster')

    def send_recommendation(self, user_id: int):
        """Отправляет рекомендацию ВСЕГДА новым сообщением."""
        print(f"Отправляю новую рекомендацию пользователю {user_id}")
        text, keyboard, poster = self.generate_recommendation_content(user_id)

        if keyboard is None:
            self.bot.send_message(user_id, text, reply_markup=None)
            return

        if poster:
            try:
                self.bot.send_photo(user_id, poster, caption=text, reply_markup=keyboard, parse_mode='HTML')
            except Exception:
                self.bot.send_message(user_id, text, reply_markup=keyboard, parse_mode='HTML')
        else:
            self.bot.send_message(user_id, text, reply_markup=keyboard, parse_mode='HTML')

    def handle_search_command(self, message):
        self.bot.send_message(message.from_user.id, "🔍 Введите описание фильма, который хотите найти...")
    
    def handle_message(self, message):
        user_id = message.from_user.id
        query = message.text.strip()
        
        if not query or len(query) < 3:
            self.bot.reply_to(message, "📝 Введите более длинный запрос (минимум 3 символа).")
            return
        
        search_msg = self.bot.reply_to(message, "🔍 Ищу фильмы по вашему запросу...")
        
        try:
            results = self.search_movies_by_description(query, top_k=5)
            if not results:
                self.bot.edit_message_text("❌ По вашему запросу ничего не найдено.", user_id, search_msg.message_id)
                return
            
            text = f"🔍 <b>Результаты по запросу «{query}»:</b>\n\n"
            for idx, (movie_id, similarity) in enumerate(results, 1):
                movie = self.db.get_movie(movie_id)
                if movie:
                    text += f"{idx}. <b>{movie['movie']}</b> ({movie.get('movie_year', '')})\n"
            
            self.bot.edit_message_text(text, user_id, search_msg.message_id, parse_mode='HTML')
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            self.bot.edit_message_text("❌ Произошла ошибка во время поиска.", user_id, search_msg.message_id)

    def search_movies_by_description(self, query: str, top_k: int = 5) -> List[tuple]:
        query_vector = self.e5_model.encode(["query: " + normalize_string(query)], normalize_embeddings=True)[0]
        all_vectors = self.db.get_all_movie_vectors()
        similarities = []
        
        for movie_data in all_vectors:
            movie_e5 = movie_data.get('vector_e5')
            if movie_e5 is not None:
                similarity = np.dot(query_vector, movie_e5)
                similarities.append((movie_data['movie_id'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def run(self):
        print("Бот запущен...")
        self.bot.polling(none_stop=True)

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        print("ОШИБКА: Телеграм токен не найден. Проверьте ваш файл config.py и переменную TELEGRAM_TOKEN")
    else:
        bot = RecommendationBot(TELEGRAM_TOKEN)
        bot.run()