from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date, datetime, timedelta
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup as bs
import logging
import requests
from typing import List, Optional



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="News Parser API",
    description="API для парсинга новостей с Lenta.ru и RBC.ru",
    version="1.0.0"
)

class Article(BaseModel):
    id: Optional[str] = Field(None, description="Уникальный идентификатор статьи")
    title: str = Field(..., description="Заголовок статьи")
    text: Optional[str] = Field(None, description="Полный текст статьи")
    date: Optional[datetime] = Field(None, description="Дата публикации статьи")
    source: str = Field(..., description="Источник статьи (Lenta.ru или RBC.ru)")

class ParserParams(BaseModel):
    date_from: date = Field(..., description="Начальная дата для парсинга (YYYY-MM-DD)")
    date_to: date = Field(..., description="Конечная дата для парсинга (YYYY-MM-DD)")
    query: str = Field(..., description="Поисковой запрос (ключевое слово)")



# --- RBC.ru Parser --- #
class RBCParser:
    def __init__(self):
        pass
    \
    def _get_url(self, param_dict: dict) -> str:
        url = 'https://www.rbc.ru/search/ajax/?' + \
        'project={0}&'.format(param_dict['project']) + \
        'category={0}&'.format(param_dict['category']) + \
        'dateFrom={0}&'.format(param_dict['dateFrom']) + \
        'dateTo={0}&'.format(param_dict['dateTo']) + \
        'page={0}&'.format(param_dict['page']) + \
        'query={0}&'.format(param_dict['query']) + \
        'material={0}'.format(param_dict['material'])
        \
        return url
    \
    def _get_article_data(self, url: str):
        try:
            r = rq.get(url)
            r.raise_for_status()
            soup = bs(r.text, features="lxml")
            \
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
            \
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
            'project': 'rbcnews',
            'category': 'TopRbcRu_economics',
            'material': '',
            'page': '0',
            'dateFrom': date_from.strftime('%d.%m.%Y'),
            'dateTo': date_to.strftime('%d.%m.%Y'),
            'query': query
        }
        
        logging.info(f"Начало парсинга для RBK.ru с date_from={date_from} по date_to={date_to}, query={query}")
        articles_df = self._iterable_load_by_page(param_copy)
            
        logging.info('Парсинг RBK.ru завершен.')
        
        return articles_df


# --- Lenta.ru Parser --- #
class LentaRuParser:
    def __init__(self):
        pass
    \
    def _get_url(self, param_dict: dict) -> str:
        hasType = int(param_dict['type']) != 0
        hasBloc = int(param_dict['bloc']) != 0

        url = 'https://lenta.ru/search/v2/process?' \
        + 'from={}&'.format(param_dict['from']) \
        + 'size={}&'.format(param_dict['size']) \
        + 'sort={}&'.format(param_dict['sort']) \
        + 'title_only={}&'.format(param_dict['title_only']) \
        + 'domain={}&'.format(param_dict['domain']) \
        + 'modified%2Cformat=yyyy-MM-dd&' \
        + 'type={}&'.format(param_dict['type']) * hasType \
        + 'bloc={}&'.format(param_dict['bloc']) * hasBloc \
        + 'modified%2Cfrom={}&'.format(param_dict['dateFrom']) \
        + 'modified%2Cto={}&'.format(param_dict['dateTo']) \
        + 'query={}'.format(param_dict['query'])
        \
        return url

    def _get_article_data(self, url: str) -> Optional[str]:
        try:
            r = rq.get(url)
            r.raise_for_status()
            soup = bs(r.text, features="lxml")
            article_body = soup.find('div', class_='topic-body')
            if article_body:
                text_parts = []
                for p_tag in article_body.find_all('p'):
                    text_parts.append(p_tag.get_text())
                text = '\n'.join(text_parts).strip()
                logging.debug(f"Текст статьи успешно получен с {url}")
                return text
            else:
                logging.warning(f"Не удалось найти тело статьи на {url}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка запроса при получении текста статьи с {url}: {e}", exc_info=True)
            return None
        except Exception as e:
            logging.error(f"Неизвестная ошибка при получении текста статьи с {url}: {e}", exc_info=True)
            return None

    def _get_search_table(self, param_dict: dict) -> pd.DataFrame:
        url = self._get_url(param_dict)
        logging.info(f"Запрос таблицы поиска для Lenta.ru: {url}")
        try:
            r = rq.get(url)
            r.raise_for_status()
            search_results = r.json().get('matches', [])
            logging.info(f"Получено {len(search_results)} результатов поиска от Lenta.ru.")
            
            articles_data = []
            for article in search_results:
                article_id = str(article.get('url'))
                date_str = article.get('modified')
                date_obj = None
                if isinstance(date_str, str):
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                    except ValueError:
                        logging.warning(f"Не удалось распарсить строковую дату '{date_str}' для статьи: {article.get('url')}")
                elif isinstance(date_str, (int, float)):
                    try:
                        date_obj = datetime.fromtimestamp(int(date_str))
                    except (ValueError, TypeError):
                        logging.warning(f"Не удалось распарсить временную метку '{date_str}' для статьи: {article.get('url')}")
                else:
                    logging.warning(f"Неожиданный тип даты '{type(date_str)}' для статьи: {article.get('url')}")
                title = article.get('title')
                article_url = article.get('url')
                
                text = self._get_article_data(article_url) if article_url else None
                
                articles_data.append({'id': article_id, 'date': date_obj, 'title': title, 'text': text, 'source': 'Lenta.ru'})
                
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
            'query'     : query, \
            'from'      : "0", # Смещение всегда 0 для API
            'size'      : "1000", # Максимальный размер, чтобы получить все за раз
            'dateFrom'  : date_from.strftime('%Y-%m-%d'),
            'dateTo'    : date_to.strftime('%Y-%m-%d'),
            'sort'      : "3", # Сортировка по дате, но Lenta.ru API может иметь свои нюансы
            'title_only': "0",
            'type'      : "0", # Все материалы
            'bloc'      : "0", # Все рубрики
            'domain'    : "1"
        }
        
        logging.info(f"Начало парсинга для Lenta.ru с date_from={date_from} по date_to={date_to}, query={query}")
        articles_df = self._get_search_table(param_copy)
            
        logging.info('Парсинг Lenta.ru завершен.')
        
        return articles_df


@app.post("/parse", response_model=List[Article])
async def parse_news(params: ParserParams):
    rbc_parser_instance = RBCParser()
    lenta_parser_instance = LentaRuParser()

    logging.info(f"Начало парсинга: date_from={params.date_from}, date_to={params.date_to}, query={params.query}")
    \
    rbc_articles_df = rbc_parser_instance.get_articles(params.date_from, params.date_to, params.query)
    lenta_articles_df = lenta_parser_instance.get_articles(params.date_from, params.date_to, params.query)

    combined_df = pd.concat([rbc_articles_df, lenta_articles_df], axis=0, ignore_index=True)
    combined_df.drop_duplicates(subset=['title', 'source'], inplace=True)
    combined_df['id'] = range(1, len(combined_df) + 1)
    combined_df['id'] = combined_df['id'].astype(str)

    logging.info(f"Завершено парсинг. Всего статей: {len(combined_df)}")
    
    output_filename = f"parsed_articles_{params.date_from}_{params.date_to}_{params.query}.csv"
    try:
        combined_df.to_csv(output_filename, index=False, encoding='utf-8')
        logging.info(f"Данные успешно сохранены в файл: {output_filename}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных в файл {output_filename}: {e}", exc_info=True)
    return combined_df.to_dict(orient='records') 