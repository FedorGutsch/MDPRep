"""
Телеграм-бот для рекомендации фильмов на основе персонального профиля пользователя
"""
import telebot
from telebot import types
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import Database
from preprocessFunc import normalize_string
from config import TELEGRAM_TOKEN
import random
import json


class RecommendationBot:
    def __init__(self, token: str, db_path: str = "movies_bot.db"):
        """Инициализация бота"""
        self.bot = telebot.TeleBot(token)
        self.db = Database(db_path)
        
        # Загрузка модели E5
        print("Загрузка модели E5...")
        self.e5_model = SentenceTransformer('intfloat/multilingual-e5-small')
        print("Модель E5 загружена!")
        
        # Инициализация TF-IDF для жанров (будет заполнена при необходимости)
        self.tfidf_vectorizer = None
        
        # Словарь для хранения состояния пользователей (калибровка)
        # {user_id: {'movies_shown': [movie_ids], 'ratings': {movie_id: rating}, 'current_movie_index': int}}
        self.user_calibration = {}
        
        self.setup_handlers()
    
    def setup_handlers(self):
        """Настройка обработчиков сообщений"""
        
        @self.bot.message_handler(commands=['start'])
        def start_handler(message):
            self.handle_start(message)
        
        @self.bot.callback_query_handler(func=lambda call: True)
        def callback_handler(call):
            self.handle_callback(call)
        
        @self.bot.message_handler(func=lambda message: True)
        def default_handler(message):
            self.handle_message(message)
    
    def handle_start(self, message):
        """Обработка команды /start"""
        user_id = message.from_user.id
        
        # Создаем пользователя если его нет
        self.db.create_user(user_id)
        
        # Получаем данные пользователя
        user = self.db.get_user(user_id)
        
        if user and user.get('calibration_complete', 0) == 1:
            # Пользователь уже прошел калибровку - предлагаем рекомендацию
            self.send_recommendation(user_id)
        else:
            # Начинаем калибровку
            self.start_calibration(user_id)
    
    def start_calibration(self, user_id: int):
        """Начать процесс калибровки"""
        # Получаем топ фильмы
        top_movies = self.db.get_top_movies(limit=250, min_rating=8.0)
        
        if len(top_movies) < 5:
            self.bot.send_message(user_id, "❌ Недостаточно фильмов в базе для калибровки.")
            return
        
        # Выбираем 5 случайных фильмов из топа
        selected_movies = random.sample(top_movies, 5)
        movie_ids = [m['movie_id'] for m in selected_movies]
        
        # Инициализируем состояние калибровки
        self.user_calibration[user_id] = {
            'movies_shown': movie_ids,
            'ratings': {},
            'current_movie_index': 0
        }
        
        # Показываем первый фильм
        self.show_calibration_movie(user_id)
    
    def show_calibration_movie(self, user_id: int):
        """Показать фильм для калибровки"""
        if user_id not in self.user_calibration:
            return
        
        calibration = self.user_calibration[user_id]
        current_idx = calibration['current_movie_index']
        
        if current_idx >= len(calibration['movies_shown']):
            # Калибровка завершена
            self.complete_calibration(user_id)
            return
        
        movie_id = calibration['movies_shown'][current_idx]
        movie = self.db.get_movie(movie_id)
        
        if not movie:
            calibration['current_movie_index'] += 1
            self.show_calibration_movie(user_id)
            return
        
        # Формируем сообщение
        text = f"🎬 <b>{movie['movie']}</b>\n\n"
        
        if movie.get('overview'):
            overview = movie['overview']
            if len(overview) > 300:
                overview = overview[:300] + "..."
            text += f"📝 {overview}\n\n"
        
        if movie.get('genres'):
            text += f"🎭 Жанры: {movie['genres']}\n"
        if movie.get('movie_year'):
            text += f"📅 Год: {movie['movie_year']}\n"
        if movie.get('kp_rating'):
            text += f"⭐ Рейтинг: {movie['kp_rating']}\n"
        
        text += f"\nОцените этот фильм (осталось {len(calibration['movies_shown']) - current_idx - 1}):"
        
        # Создаем кнопки
        keyboard = types.InlineKeyboardMarkup(row_width=3)
        buttons = [
            types.InlineKeyboardButton("⭐ 1", callback_data=f"rate_{movie_id}_1"),
            types.InlineKeyboardButton("⭐ 2", callback_data=f"rate_{movie_id}_2"),
            types.InlineKeyboardButton("⭐ 3", callback_data=f"rate_{movie_id}_3"),
            types.InlineKeyboardButton("⭐ 4", callback_data=f"rate_{movie_id}_4"),
            types.InlineKeyboardButton("⭐ 5", callback_data=f"rate_{movie_id}_5"),
            types.InlineKeyboardButton("❌ Не смотрел", callback_data=f"rate_{movie_id}_skip")
        ]
        keyboard.add(*buttons)
        
        self.bot.send_message(user_id, text, reply_markup=keyboard, parse_mode='HTML')
    
    def handle_callback(self, call):
        """Обработка callback-запросов (нажатия на кнопки)"""
        user_id = call.from_user.id
        
        if call.data.startswith("rate_"):
            # Обработка оценки фильма
            parts = call.data.split("_")
            movie_id = int(parts[1])
            rating = parts[2]
            
            if user_id not in self.user_calibration:
                self.bot.answer_callback_query(call.id, "Ошибка: калибровка не начата")
                return
            
            calibration = self.user_calibration[user_id]
            
            if rating == "skip":
                # Пропускаем фильм
                self.bot.answer_callback_query(call.id, "Фильм пропущен")
            else:
                # Сохраняем оценку
                calibration['ratings'][movie_id] = int(rating)
                self.bot.answer_callback_query(call.id, f"Оценка {rating} сохранена!")
            
            # Переходим к следующему фильму
            calibration['current_movie_index'] += 1
            
            # Удаляем сообщение
            try:
                self.bot.delete_message(user_id, call.message.message_id)
            except:
                pass
            
            # Показываем следующий фильм или завершаем калибровку
            self.show_calibration_movie(user_id)
        
        elif call.data == "get_recommendation":
            # Запрос новой рекомендации
            self.send_recommendation(user_id)
            self.bot.answer_callback_query(call.id)
    
    def complete_calibration(self, user_id: int):
        """Завершить калибровку и вычислить персональный вектор"""
        if user_id not in self.user_calibration:
            return
        
        calibration = self.user_calibration[user_id]
        ratings = calibration['ratings']
        
        if len(ratings) == 0:
            self.bot.send_message(user_id, 
                "❌ Вы не оценили ни одного фильма. Начните заново командой /start")
            del self.user_calibration[user_id]
            return
        
        # Вычисляем персональный вектор пользователя
        print(f"Вычисление персонального вектора для пользователя {user_id}...")
        
        # Собираем векторы оцененных фильмов
        user_e5_vectors = []
        user_tfidf_vectors = []
        user_combined_vectors = []
        weights = []
        
        for movie_id, rating in ratings.items():
            movie_vector = self.db.get_movie_vector(movie_id)
            if movie_vector:
                # Используем оценку как вес (нормализуем от 1-5 до 0.2-1.0)
                weight = (rating - 1) / 4.0  # 1 -> 0.0, 5 -> 1.0
                weight = weight * 0.8 + 0.2  # Сдвигаем в диапазон 0.2-1.0
                
                weights.append(weight)
                user_e5_vectors.append(movie_vector['vector_e5'])
                user_tfidf_vectors.append(movie_vector['vector_tfidf'])
                user_combined_vectors.append(movie_vector['combined_vector'])
        
        if len(user_e5_vectors) == 0:
            self.bot.send_message(user_id, 
                "❌ Ошибка при вычислении вектора. Попробуйте еще раз.")
            del self.user_calibration[user_id]
            return
        
        # Вычисляем взвешенное среднее
        weights = np.array(weights)
        weights = weights / weights.sum()  # Нормализуем веса
        
        user_e5 = np.average(user_e5_vectors, axis=0, weights=weights)
        user_tfidf = np.average(user_tfidf_vectors, axis=0, weights=weights)
        user_combined = np.average(user_combined_vectors, axis=0, weights=weights)
        
        # Нормализуем векторы
        if np.linalg.norm(user_e5) > 0:
            user_e5 = user_e5 / np.linalg.norm(user_e5)
        if np.linalg.norm(user_tfidf) > 0:
            user_tfidf = user_tfidf / np.linalg.norm(user_tfidf)
        if np.linalg.norm(user_combined) > 0:
            user_combined = user_combined / np.linalg.norm(user_combined)
        
        # Сохраняем вектор пользователя
        self.db.update_user_vector(
            user_id=user_id,
            vector_e5=user_e5,
            vector_tfidf=user_tfidf,
            combined_vector=user_combined
        )
        
        # Отмечаем калибровку как завершенную
        self.db.set_calibration_complete(user_id, complete=True)
        
        # Удаляем состояние калибровки
        del self.user_calibration[user_id]
        
        # Отправляем сообщение о завершении калибровки
        self.bot.send_message(user_id, 
            "✅ Калибровка завершена! Ваш персональный профиль создан.\n\n"
            "Теперь я могу рекомендовать фильмы специально для вас!")
        
        # Предлагаем первую рекомендацию
        self.send_recommendation(user_id)
    
    def send_recommendation(self, user_id: int):
        """Отправить рекомендацию фильма пользователю"""
        user = self.db.get_user(user_id)
        
        if not user or user.get('combined_vector') is None:
            self.bot.send_message(user_id, 
                "❌ Ваш профиль еще не создан. Начните калибровку командой /start")
            return
        
        user_vector = user['combined_vector']
        
        # Получаем все векторы фильмов
        all_movie_ids = self.db.get_all_movie_ids()
        
        if len(all_movie_ids) == 0:
            self.bot.send_message(user_id, "❌ В базе нет фильмов для рекомендации.")
            return
        
        # Вычисляем схожесть с вектором пользователя
        similarities = []
        
        for movie_id in all_movie_ids:
            movie_vector_data = self.db.get_movie_vector(movie_id)
            if movie_vector_data and movie_vector_data.get('combined_vector') is not None:
                movie_vector = movie_vector_data['combined_vector']
                
                # Косинусное сходство
                similarity = np.dot(user_vector, movie_vector)
                similarities.append((movie_id, similarity))
        
        if len(similarities) == 0:
            self.bot.send_message(user_id, "❌ Не удалось найти подходящие фильмы.")
            return
        
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Берем топ-5 самых похожих фильмов и выбираем случайный из них
        # (чтобы было разнообразие)
        top_n = min(5, len(similarities))
        top_movies = similarities[:top_n]
        
        # Выбираем случайный из топ-5
        recommended_movie_id, similarity_score = random.choice(top_movies)
        
        # Получаем информацию о фильме
        movie = self.db.get_movie(recommended_movie_id)
        
        if not movie:
            self.bot.send_message(user_id, "❌ Ошибка при получении информации о фильме.")
            return
        
        # Формируем сообщение
        text = f"🎬 <b>Рекомендация для вас:</b>\n\n"
        text += f"<b>{movie['movie']}</b>\n\n"
        
        if movie.get('overview'):
            overview = movie['overview']
            if len(overview) > 400:
                overview = overview[:400] + "..."
            text += f"📝 {overview}\n\n"
        
        if movie.get('genres'):
            text += f"🎭 Жанры: {movie['genres']}\n"
        if movie.get('movie_year'):
            text += f"📅 Год: {movie['movie_year']}\n"
        if movie.get('kp_rating'):
            text += f"⭐ Рейтинг Кинопоиска: {movie['kp_rating']}\n"
        if movie.get('movie_duration'):
            text += f"⏱ Длительность: {movie['movie_duration']} мин\n"
        
        text += f"\n💡 Совпадение с вашими предпочтениями: {similarity_score:.2%}"
        
        # Создаем кнопку для новой рекомендации
        keyboard = types.InlineKeyboardMarkup()
        keyboard.add(types.InlineKeyboardButton("🎲 Еще одну рекомендацию", 
                                               callback_data="get_recommendation"))
        
        # Отправляем постер, если есть
        if movie.get('poster'):
            try:
                self.bot.send_photo(user_id, movie['poster'], caption=text, 
                                  reply_markup=keyboard, parse_mode='HTML')
            except:
                # Если не удалось отправить фото, отправляем текст
                self.bot.send_message(user_id, text, reply_markup=keyboard, 
                                    parse_mode='HTML')
        else:
            self.bot.send_message(user_id, text, reply_markup=keyboard, 
                                parse_mode='HTML')
    
    def handle_message(self, message):
        """Обработка обычных сообщений"""
        user_id = message.from_user.id
        user = self.db.get_user(user_id)
        
        if user and user.get('calibration_complete', 0) == 1:
            # Если калибровка завершена, предлагаем рекомендацию
            self.send_recommendation(user_id)
        else:
            # Если калибровка не завершена, напоминаем начать
            self.bot.reply_to(message, 
                "👋 Для начала работы с ботом используйте команду /start")
    
    def run(self):
        """Запустить бота"""
        print("Бот запущен...")
        self.bot.polling(none_stop=True)


if __name__ == "__main__":
    bot = RecommendationBot(TELEGRAM_TOKEN)
    bot.run()

