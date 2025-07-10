from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date, datetime, timedelta
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup as bs
import logging
import requests
from typing import List, Optional

# --- Импорты и настройка Natasha ---
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)

# --- Импорт для анализа тональности ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from NoDuplicates import deduplicate_news_with_annoy
from okved_analyzer import OKVEDAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Глобальная инициализация моделей Natasha для производительности ---
logging.info("Загрузка моделей Natasha...")
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
logging.info("Модели Natasha успешно загружены.")

# --- Глобальная инициализация модели для анализа тональности ---
logging.info("Загрузка модели для анализа тональности...")
try:
    # Используем модель для анализа тональности на русском языке
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="blanchefort/rubert-base-cased-sentiment-rusentiment",
        tokenizer="blanchefort/rubert-base-cased-sentiment-rusentiment"
    )
    logging.info("Модель тональности успешно загружена.")
except Exception as e:
    logging.warning(f"Не удалось загрузить модель тональности: {e}. Будет использована заглушка.")
    sentiment_pipeline = None

# --- Инициализация анализатора ОКВЭД ---
logging.info("Инициализация анализатора ОКВЭД...")
try:
    okved_analyzer = OKVEDAnalyzer()
    logging.info("Анализатор ОКВЭД успешно инициализирован.")
except Exception as e:
    logging.error(f"Ошибка при инициализации анализатора ОКВЭД: {e}")
    okved_analyzer = None
# --- Конец блока инициализации ---

app = FastAPI(
    title="News Parser API",
    description="API для парсинга новостей с RBC.ru с функциями: извлечения названий компаний, анализа тональности, дедупликации и определения кодов ОКВЭД.",
    version="2.0.0"
)

class Article(BaseModel):
    id: Optional[str] = Field(None, description="Уникальный идентификатор статьи")
    title: str = Field(..., description="Заголовок статьи")
    text: Optional[str] = Field(None, description="Полный текст статьи")
    date: Optional[datetime] = Field(None, description="Дата публикации статьи")
    source: str = Field(..., description="Источник статьи (Lenta.ru или RBC.ru)")
    found_companies: Optional[List[str]] = Field(None, description="Список компаний, найденных в тексте статьи")
    sentiment: Optional[str] = Field(None, description="Тональность новости (положительная, отрицательная, нейтральная)")
    sentiment_score: Optional[float] = Field(None, description="Числовая оценка тональности от 0 до 1")
    okved_codes: Optional[str] = Field(None, description="Коды ОКВЭД через запятую")
    okved_descriptions: Optional[str] = Field(None, description="Описания кодов ОКВЭД через запятую")
    okved_scores: Optional[str] = Field(None, description="Оценки соответствия кодов ОКВЭД через запятую")

class ParserParams(BaseModel):
    date_from: date = Field(..., description="Начальная дата для парсинга (YYYY-MM-DD)")
    date_to: date = Field(..., description="Конечная дата для парсинга (YYYY-MM-DD)")
    query: str = Field(..., description="Поисковой запрос (ключевое слово)")

def extract_companies_ner(text: Optional[str]) -> List[str]:
    """
    Извлекает названия организаций из текста с помощью Natasha.
    Возвращает пустой список, если текст отсутствует или компании не найдены.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)

    found_companies = set()
    for span in doc.spans:
        if span.type == 'ORG':
            span.normalize(morph_vocab)
            found_companies.add(span.normal)
            
    return list(found_companies)

def analyze_sentiment_for_companies(text: Optional[str], companies: List[str]) -> tuple[str, float]:
    """
    Анализирует тональность текста с фокусом на упоминания компаний и экономические показатели.
    """
    if not isinstance(text, str) or not text.strip() or not companies:
        return "нейтральная", 0.5
    
    if sentiment_pipeline is None:
        return "нейтральная", 0.5
    
    try:
        # Экономические индикаторы
        positive_indicators = ['рост', 'увеличение', 'прибыль', 'успех', 'развитие', 'расширение', 
                             'инвестиции', 'сделка', 'контракт', 'партнерство']
        negative_indicators = ['падение', 'снижение', 'убыток', 'банкротство', 'сокращение', 
                             'закрытие', 'штраф', 'санкции', 'проблемы', 'риски']
        
        # Разбиваем текст на предложения с упоминанием компаний
        relevant_sentences = []
        sentences = text.split('.')
        for sentence in sentences:
            if any(company.lower() in sentence.lower() for company in companies):
                relevant_sentences.append(sentence)
        
        if not relevant_sentences:
            return "нейтральная", 0.5
            
        # Анализируем каждое релевантное предложение
        sentiments = []
        for sentence in relevant_sentences:
            # Проверяем наличие экономических индикаторов
            pos_count = sum(1 for indicator in positive_indicators if indicator in sentence.lower())
            neg_count = sum(1 for indicator in negative_indicators if indicator in sentence.lower())
            
            # Получаем базовую тональность от модели
            result = sentiment_pipeline(sentence[:512])[0]
            base_score = result['score']
            
            # Корректируем оценку с учетом экономических индикаторов
            adjusted_score = base_score
            if pos_count > neg_count:
                adjusted_score = min(1.0, base_score + 0.2)
            elif neg_count > pos_count:
                adjusted_score = max(0.0, base_score - 0.2)
                
            sentiments.append(adjusted_score)
        
        # Вычисляем среднюю оценку
        final_score = sum(sentiments) / len(sentiments)
        
        # Определяем итоговую тональность
        if final_score >= 0.6:
            sentiment = "положительная"
        elif final_score <= 0.4:
            sentiment = "отрицательная"
        else:
            sentiment = "нейтральная"
            
        return sentiment, final_score
        
    except Exception as e:
        logging.error(f"Ошибка при анализе тональности: {e}")
        return "нейтральная", 0.5

# --- RBC.ru Parser --- #
class RBCParser:
    def __init__(self):
        pass
    
    def _get_url(self, param_dict: dict) -> str:
        url = 'https://www.rbc.ru/search/ajax/?' + \
        'project={0}&'.format(param_dict['project']) + \
        'dateFrom={0}&'.format(param_dict['dateFrom']) + \
        'dateTo={0}&'.format(param_dict['dateTo']) + \
        'page={0}&'.format(param_dict['page']) + \
        'query={0}&'.format(param_dict['query'])
        return url
    
    def _get_article_data(self, url: str):
        try:
            r = rq.get(url)
            r.raise_for_status()
            soup = bs(r.text, features="lxml")
            
            div_overview = soup.find('div', {'class': 'article__text__overview'})
            overview = div_overview.text.replace('<br />', '\n').strip() if div_overview else None

            article_body = soup.find('div', class_='article__text')
            if article_body:
                for tag in article_body.find_all(True):
                    if tag.name != 'p':
                        tag.decompose()
                text = '\n'.join([p.get_text() for p in article_body.find_all('p')])
                text = text.strip()
                logging.debug(f"Текст статьи успешно получен с {url}")
            else:
                text = None
                logging.warning(f"Не удалось найти тело статьи на {url}")
            
            return overview, text
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка запроса при получении текста статьи с {url}: {e}", exc_info=True)
            return None, None
        except Exception as e:
            logging.error(f"Неизвестная ошибка при получении текста статьи с {url}: {e}", exc_info=True)
            return None, None

    def _iterable_load_by_page(self, param_dict):
        param_copy = param_dict.copy()
        results = []
        current_page = int(param_copy.get('page', 0))

        while True:
            param_copy['page'] = str(current_page)
            result_df = self._get_search_table(param_copy)
            if result_df.empty:
                break
            results.append(result_df)
            current_page += 1
            
        if results:
            return pd.concat(results, axis=0, ignore_index=True)
        return pd.DataFrame()

    def _get_search_table(self, param_dict: dict) -> pd.DataFrame:
        url = self._get_url(param_dict)
        logging.info(f"Запрос таблицы поиска для RBK.ru: {url}")
        try:
            r = rq.get(url)
            r.raise_for_status()
            search_results = r.json().get('items', [])
            logging.info(f"Получено {len(search_results)} результатов поиска от RBK.ru.")
            
            articles_data = []
            for article in search_results:
                article_id = str(article.get('id')) if article.get('id') is not None else None
                date_ts = article.get('publish_date_t')
                date_obj = datetime.fromtimestamp(date_ts) if date_ts else None
                title = article.get('title')
                article_url = article.get('fronturl')
                
                overview, text = self._get_article_data(article_url) if article_url else (None, None)
                
                articles_data.append({'id': article_id, 'date': date_obj, 'title': title, 'text': text, 'source': 'RBC.ru'})
                
            return pd.DataFrame(articles_data)
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка запроса при получении таблицы поиска с {url}: {e}", exc_info=True)
            return pd.DataFrame(columns=['id', 'date', 'title', 'text', 'source'])
        except Exception as e:
            logging.error(f"Неизвестная ошибка при получении таблицы поиска с {url}: {e}", exc_info=True)
            return pd.DataFrame(columns=['id', 'date', 'title', 'text', 'source'])

    def get_articles(self,
                        date_from: date,
                        date_to: date,
                        query: str) -> pd.DataFrame:
        param_copy = {
            'project': 'quote',
            'page': '0',
            'dateFrom': date_from.strftime('%d.%m.%Y'),
            'dateTo': date_to.strftime('%d.%m.%Y'),
            'query': query
        }
        
        logging.info(f"Начало парсинга для RBK.ru с date_from={date_from} по date_to={date_to}, query={query}")
        articles_df = self._iterable_load_by_page(param_copy)
            
        logging.info('Парсинг RBK.ru завершен.')
        
        return articles_df



@app.post("/parse", response_model=List[Article])
async def parse_news(params: ParserParams):
    rbc_parser_instance = RBCParser()

    logging.info(f"Начало парсинга: date_from={params.date_from}, date_to={params.date_to}, query={params.query}")
    
    rbc_articles_df = rbc_parser_instance.get_articles(params.date_from, params.date_to, params.query)
    combined_df = rbc_articles_df.copy() 
    
    logging.info("Начало дедупликации новостей...")
    articles_list_for_dedup = combined_df.to_dict(orient='records')
    cleaned_articles_list = deduplicate_news_with_annoy(articles_list_for_dedup)
    combined_df = pd.DataFrame(cleaned_articles_list)
    logging.info(f"Дедупликация завершена. Осталось статей после дедупликации: {len(combined_df)}")

    if len(combined_df) >= 1:
        combined_df['id'] = range(1, len(combined_df) + 1)
        combined_df['id'] = combined_df['id'].astype(str)
        
        # Извлечение компаний
        logging.info("Начало извлечения названий компаний...")
        combined_df['found_companies'] = combined_df['text'].apply(extract_companies_ner)
        logging.info("Извлечение названий компаний завершено.")
        
        # Анализ тональности с учетом найденных компаний
        logging.info("Начало анализа тональности...")
        sentiment_results = combined_df.apply(
            lambda row: analyze_sentiment_for_companies(row['text'], row['found_companies']), 
            axis=1
        )
        combined_df['sentiment'] = sentiment_results.apply(lambda x: x[0])
        combined_df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])
        logging.info("Анализ тональности завершен.")
        
        # Анализ ОКВЭД
        if okved_analyzer is not None:
            logging.info("Начало анализа ОКВЭД...")
            # Получаем коды ОКВЭД
            okved_results = combined_df.apply(
                lambda row: okved_analyzer.find_relevant_okved_codes(
                    row['text'], 
                    row['title'],
                    top_k=10
                ), 
                axis=1
            )
            # Преобразуем результаты в строки
            combined_df['okved_codes'] = okved_results.apply(
                lambda codes: ', '.join([code['okved_code'] for code in codes]) if codes else ''
            )
            combined_df['okved_descriptions'] = okved_results.apply(
                lambda codes: ', '.join([code['okved_description'] for code in codes]) if codes else ''
            )
            combined_df['okved_scores'] = okved_results.apply(
                lambda codes: ', '.join([str(code['similarity_score']) for code in codes]) if codes else ''
            )
            logging.info("Анализ ОКВЭД завершен.")
        else:
            combined_df['okved_codes'] = ''
            combined_df['okved_descriptions'] = ''
            combined_df['okved_scores'] = ''

        logging.info(f"Обработка завершена. Всего статей: {len(combined_df)}")
        
        output_filename = f"parsed_articles_{params.date_from}_{params.date_to}_{params.query}.csv"
        # try:
            # combined_df.to_csv(output_filename, index=False, encoding='utf-8')
            # logging.info(f"Данные успешно сохранены в файл: {output_filename}")
        # except Exception as e:
            # logging.error(f"Ошибка при сохранении данных в файл {output_filename}: {e}", exc_info=True)
        
        return combined_df.to_dict(orient='records')
    return []