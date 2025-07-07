import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
from typing import List, Dict, Tuple
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OKVEDAnalyzer:
    """
    Класс для анализа соответствия новостей кодам ОКВЭД
    """
    
    def __init__(self, okved_csv_path: str = "okved_codes.csv", similarity_threshold: float = 0.3):
        """
        Инициализация анализатора ОКВЭД
        
        Args:
            okved_csv_path: путь к файлу с кодами ОКВЭД
            similarity_threshold: порог схожести для определения соответствия
        """
        self.okved_csv_path = okved_csv_path
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.okved_df = None
        self.okved_embeddings = None
        
        self._load_okved_data()
        self._load_model()
        
    def _load_okved_data(self):
        """Загружает данные ОКВЭД из CSV файла"""
        try:
            self.okved_df = pd.read_csv(self.okved_csv_path)
            # Очищаем описания от лишних пробелов
            self.okved_df['description'] = self.okved_df['description'].str.strip()
            logging.info(f"Загружено {len(self.okved_df)} кодов ОКВЭД")
        except Exception as e:
            logging.error(f"Ошибка при загрузке файла ОКВЭД: {e}")
            raise
            
    def _load_model(self):
        """Загружает модель для векторизации текста"""
        try:
            logging.info("Загрузка модели для анализа ОКВЭД...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logging.info("Модель для анализа ОКВЭД успешно загружена")
            
            # Предварительно векторизуем все описания ОКВЭД
            logging.info("Векторизация описаний ОКВЭД...")
            okved_texts = self.okved_df['description'].tolist()
            self.okved_embeddings = self.model.encode(okved_texts, show_progress_bar=True)
            logging.info("Векторизация описаний ОКВЭД завершена")
            
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def _clean_text_for_analysis(self, text: str) -> str:
        """Очищает текст для анализа"""
        if not isinstance(text, str):
            return ""
        
        # Удаляем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ограничиваем длину текста (берем первые 1000 символов)
        if len(text) > 1000:
            text = text[:1000]
            
        return text
    
    def find_relevant_okved_codes(self, news_text: str, news_title: str = "", 
                                  top_k: int = 10) -> List[Dict]:
        """
        Находит релевантные коды ОКВЭД для новости
        
        Args:
            news_text: текст новости
            news_title: заголовок новости
            top_k: количество наиболее релевантных кодов для возврата (по умолчанию 10)
            
        Returns:
            Список словарей с кодами ОКВЭД и их оценками соответствия
        """
        if self.model is None or self.okved_embeddings is None:
            logging.error("Модель или эмбеддинги ОКВЭД не загружены")
            return []
        
        # Объединяем заголовок и текст новости
        combined_text = f"{news_title}. {news_text}".strip()
        combined_text = self._clean_text_for_analysis(combined_text)
        
        if not combined_text:
            return []
        
        try:
            # Векторизуем текст новости
            news_embedding = self.model.encode([combined_text])
            
            # Вычисляем схожесть с каждым описанием ОКВЭД
            similarities = util.cos_sim(news_embedding, self.okved_embeddings)[0]
            
            # Получаем индексы наиболее похожих описаний
            similarities_np = similarities.cpu().numpy()
            top_indices = np.argsort(similarities_np)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                similarity_score = similarities[idx].item()
                
                # Применяем порог схожести
                if similarity_score >= self.similarity_threshold:
                    okved_code = self.okved_df.iloc[idx]['code']
                    okved_description = self.okved_df.iloc[idx]['description']
                    
                    results.append({
                        'okved_code': str(okved_code),
                        'okved_description': okved_description,
                        'similarity_score': round(similarity_score, 4)
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Ошибка при поиске кодов ОКВЭД: {e}")
            return []
    
    def analyze_news_csv(self, input_csv_path: str, output_csv_path: str = None) -> pd.DataFrame:
        """
        Анализирует CSV файл с новостями и добавляет соответствующие коды ОКВЭД
        
        Args:
            input_csv_path: путь к входному CSV файлу с новостями
            output_csv_path: путь к выходному CSV файлу (если None, генерируется автоматически)
            
        Returns:
            DataFrame с результатами анализа
        """
        try:
            # Загружаем новости
            news_df = pd.read_csv(input_csv_path)
            logging.info(f"Загружено {len(news_df)} новостей для анализа")
            
            # Проверяем наличие необходимых колонок
            required_columns = ['title', 'text']
            missing_columns = [col for col in required_columns if col not in news_df.columns]
            if missing_columns:
                raise ValueError(f"В файле отсутствуют необходимые колонки: {missing_columns}")
            
            # Анализируем каждую новость
            okved_results = []
            for idx, row in news_df.iterrows():
                logging.info(f"Анализ новости {idx + 1}/{len(news_df)}")
                
                title = str(row.get('title', ''))
                text = str(row.get('text', ''))
                
                relevant_codes = self.find_relevant_okved_codes(text, title)
                
                if relevant_codes:
                    for code_info in relevant_codes:
                        okved_results.append({
                            'news_id': idx,
                            'news_title': title,
                            'okved_code': code_info['okved_code'],
                            'okved_description': code_info['okved_description'],
                            'similarity_score': code_info['similarity_score']
                        })
                else:
                    # Если коды не найдены, добавляем запись с пустыми значениями
                    okved_results.append({
                        'news_id': idx,
                        'news_title': title,
                        'okved_code': '',
                        'okved_description': 'Соответствующие коды ОКВЭД не найдены',
                        'similarity_score': 0.0
                    })
            
            # Создаем DataFrame с результатами
            results_df = pd.DataFrame(okved_results)
            
            # Сохраняем результаты
            if output_csv_path is None:
                output_csv_path = input_csv_path.replace('.csv', '_okved_analysis.csv')
            
            results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            logging.info(f"Результаты анализа ОКВЭД сохранены в файл: {output_csv_path}")
            
            return results_df
            
        except Exception as e:
            logging.error(f"Ошибка при анализе новостей: {e}")
            raise


def main():
    """
    Основная функция для запуска анализа из командной строки
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python okved_analyzer.py <путь_к_csv_с_новостями>")
        print("Опционально: python okved_analyzer.py <путь_к_csv_с_новостями> <путь_к_выходному_файлу>")
        return
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        analyzer = OKVEDAnalyzer()
        results = analyzer.analyze_news_csv(input_csv, output_csv)
        
        print(f"\nАнализ завершен!")
        print(f"Обработано новостей: {len(results['news_id'].unique())}")
        print(f"Найдено соответствий ОКВЭД: {len(results[results['okved_code'] != ''])}")
        print(f"Результаты сохранены в: {output_csv or input_csv.replace('.csv', '_okved_analysis.csv')}")
        
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main() 