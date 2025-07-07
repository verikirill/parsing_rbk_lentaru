import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from annoy import AnnoyIndex

def deduplicate_news_with_annoy(articles: list, threshold: float = 0.9) -> list:
    if not articles:
        return []

    print("Шаг 1: Загрузка модели и векторизация новостей...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    texts_to_vectorize = []
    for article in articles:
        text = article.get('text')
        title = article.get('title')
        if title and text:
            texts_to_vectorize.append(f"{title}. {text}")
        elif title:
            texts_to_vectorize.append(title)
        elif text:
            texts_to_vectorize.append(text)
        else:
            texts_to_vectorize.append("") # Добавляем пустую строку, чтобы сохранить соответствие индексов

    if not texts_to_vectorize:
        return articles # Возвращаем оригинальные статьи, если нет текста для векторизации

    embeddings = model.encode(texts_to_vectorize, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    print(f"Векторизация завершена. Размерность вектора: {dimension}")

    print("\nШаг 2: Создание индекса Annoy и поиск дубликатов...")
    annoy_index = AnnoyIndex(dimension, 'angular')
    for i, vector in enumerate(embeddings):
        annoy_index.add_item(i, vector)

    annoy_index.build(10) 
    print("Индекс Annoy построен.")

    duplicate_indices = set()
    
    for i in range(len(articles)):
        if i in duplicate_indices:
            continue
        neighbors_indices, neighbor_distances = annoy_index.get_nns_by_item(i, 5, include_distances=True)

        print(f"\nАнализ новости #{i}: '{articles[i]['title'][:50]}...'")
        print(f"Найдено соседей: {len(neighbors_indices)}")
        for j in range(1, len(neighbors_indices)):
            neighbor_idx = neighbors_indices[j]
            similarity = util.cos_sim(embeddings[i], embeddings[neighbor_idx]).item()
            
            print(f"  Сравнение с новостью #{neighbor_idx}: '{articles[neighbor_idx]['title'][:50]}...'")
            print(f"  Схожесть: {similarity:.4f} (порог: {threshold})")

            if similarity > threshold:
                print(f"  → ДУБЛИКАТ НАЙДЕН!")
                duplicate_indices.add(neighbor_idx)
            else:
                print(f"  → Не дубликат (схожесть {similarity:.4f} < {threshold})")
    
    print(f"\nНайдено дубликатов: {len(duplicate_indices)}")
    cleaned_articles = [article for idx, article in enumerate(articles) if idx not in duplicate_indices]
    
    return cleaned_articles