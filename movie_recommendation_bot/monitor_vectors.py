# Файл: monitor_vectors.py (ВСЕГДА строит графики)

import argparse
import sqlite3
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Используем бэкенд, который не требует GUI
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def fetch_users(db_path: str):
    """Получить всех пользователей с завершенной калибровкой."""
    if not os.path.exists(db_path): return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, vector_e5, vector_tfidf, combined_vector, ratings_count
        FROM users 
        WHERE calibration_complete = 1 AND combined_vector IS NOT NULL
        ORDER BY user_id
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def print_snapshot(db_path: str):
    """Вывести информативный снимок пользователей."""
    users = fetch_users(db_path)
    if not users:
        print("--- МОНИТОРИНГ ПОЛЬЗОВАТЕЛЕЙ ---")
        print("\nПока нет пользователей, завершивших калибровку.")
        print("-> Запустите бота, пройдите калибровку (/start) или запустите populate_test_users.py")
        return False

    print(f"\n{'='*60}")
    print(f"📊 МОНИТОРИНГ ПОЛЬЗОВАТЕЛЕЙ ({len(users)} чел.)")
    print(f"{'='*60}")
    
    for row in users:
        print(f"\n👤 User ID: {row['user_id']} (Оценок: {row['ratings_count']})")
        
        try:
            if row["vector_e5"]:
                vec_e5 = np.array(json.loads(row["vector_e5"]))
                print(f"   📐 E5 вектор:       mean={np.mean(vec_e5):.3f}, std={np.std(vec_e5):.3f}, dim={len(vec_e5)}")
            
            if row["vector_tfidf"]:
                vec_tfidf = np.array(json.loads(row["vector_tfidf"]))
                print(f"   📐 TF-IDF вектор:   mean={np.mean(vec_tfidf):.3f}, std={np.std(vec_tfidf):.3f}, dim={len(vec_tfidf)}")
            
            if row["combined_vector"]:
                vec_combined = np.array(json.loads(row["combined_vector"]))
                print(f"   📐 Combined вектор: mean={np.mean(vec_combined):.3f}, std={np.std(vec_combined):.3f}, dim={len(vec_combined)}")
        except Exception as e:
            print(f"   ❌ Не удалось проанализировать векторы для пользователя {row['user_id']}. Ошибка: {e}")
    
    print(f"\n{'='*60}")
    return True

def plot_user_vectors(db_path: str, output_file: str = "user_vectors_plot.png"):
    """Создать график распределения пользователей."""
    print(f"\n-> Попытка построить график векторов...")
    users = fetch_users(db_path)
    if len(users) < 2:
        print("-> Недостаточно пользователей для построения графика (нужно минимум 2).")
        return
    
    vectors, user_ids = [], []
    for row in users:
        if row["combined_vector"]:
            try:
                vectors.append(np.array(json.loads(row["combined_vector"])))
                user_ids.append(row["user_id"])
            except: continue
    
    if len(vectors) < 2:
        print("-> Недостаточно валидных векторов для графика.")
        return

    vectors = np.array(vectors)
    
    # Снижение размерности
    if len(vectors) > 3:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
        vectors_2d = tsne.fit_transform(vectors)
    else:
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=range(len(user_ids)), cmap='viridis', s=100)
    for i, user_id in enumerate(user_ids):
        plt.annotate(f'U{user_id}', (vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.colorbar(scatter, label='Индекс пользователя')
    plt.title('Распределение Пользователей в Пространстве Вкусов (t-SNE/PCA)')
    plt.xlabel('Компонента 1'); plt.ylabel('Компонента 2')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=120)
    plt.close()
    print(f"✅ График распределения пользователей сохранен в файл: {output_file}")

def main():

  
    while True:
        try:
            if print_snapshot('movies_bot.db'):
                plot_user_vectors('movies_bot.db')
            print(f"\nСледующее обновление через 10 секунд (Нажмите Ctrl+C для выхода)")
            time.sleep(10000)
        except KeyboardInterrupt:
            print("\n\n⏹ Мониторинг остановлен.")
            break

if __name__ == "__main__":
    main()