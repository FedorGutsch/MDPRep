import telebot
from config import TELEGRAM_TOKEN  # Добавьте ваш токен в config.py

# Ваш существующий код
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd 
import pymorphy3 as pm

   
STOP_WORDS = {'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она',
'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее',
'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до',
'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем',
'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним',
'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда',
'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая',
'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед',
'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно',
'всю', 'между', 'фильм'}
     

def find_similar_movies(user_query, source, vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))):
    morph = pm.MorphAnalyzer()
    normalized = pd.read_csv(source)

    
    tfidf_matrix = vectorizer.fit_transform(normalized['overview'])


    s = ' '
    # Нормализация запроса
    for i in user_query.split():
        if i.lower() not in STOP_WORDS:  # Удаляем стоп-слова
            s += morph.parse(i)[0].normal_form + ' '

    # Поиск похожих фильмов
    query_vector = vectorizer.transform([s])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    similarity_scores = cosine_similarities[0]

    non_zero_indices = np.where(similarity_scores > 0.1)[0]
    non_zero_scores = similarity_scores[non_zero_indices]

    response = ''
    if len(non_zero_indices) > 0:
        sorted_indices = non_zero_indices[np.argsort(non_zero_scores)[::-1]]
        top_indices = sorted_indices[:1]
        top_scores = similarity_scores[top_indices]

        for i, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
            best_match_title = normalized.loc[idx, 'movie']
            response += f"{best_match_title}"

    else:
        response = " "
        
    return response


def find_similar_by_fasttext(user_query, ft_model, movie_vectors_matrix, movies_df, morph, similarity_threshold=0.5):
    """
    Находит похожий фильм, используя предобученную модель FastText и матрицу векторов.
    
    :param user_query: Текстовый запрос пользователя.
    :param ft_model: Загруженная модель FastText.
    :param movie_vectors_matrix: NumPy матрица с векторами для каждого фильма.
    :param movies_df: DataFrame с данными о фильмах (нужна колонка 'movie').
    :param morph: Инициализированный объект pymorphy3.MorphAnalyzer.
    :param similarity_threshold: Порог сходства для отсеивания неподходящих результатов.
    :return: Строка с названием самого похожего фильма или None, если ничего не найдено.
    """
    
    # 1. Нормализация запроса пользователя
    processed_query = []
    for word in user_query.lower().split():
        if word not in STOP_WORDS:
            processed_query.append(morph.parse(word)[0].normal_form)
    
    if not processed_query:
        return None # Возвращаем None, если запрос состоял только из стоп-слов

    final_query = " ".join(processed_query)

    # 2. Получение вектора для запроса с помощью модели FastText
    query_vector = ft_model.get_sentence_vector(final_query)

    # 3. Вычисление косинусного сходства
    # query_vector нужно преобразовать в (1, N) матрицу для функции cosine_similarity
    cosine_similarities = cosine_similarity(query_vector.reshape(1, -1), movie_vectors_matrix)
    
    # cosine_similarities - это матрица (1, M), где M - кол-во фильмов. Берем первую строку.
    similarity_scores = cosine_similarities[0]

    # 4. Поиск лучшего совпадения
    best_match_index = np.argmax(similarity_scores)
    best_score = similarity_scores[best_match_index]

    if best_score > similarity_threshold:
        best_match_title = movies_df.loc[best_match_index, 'movie']
        return best_match_title
    else:
        return None