import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from IPython import display



class rbc_parser:
    def __init__(self):
        pass
    
    
    def _get_url(self, param_dict: dict) -> str:
        """
        Возвращает URL для запроса json таблицы со статьями
        """
        url = 'https://www.rbc.ru/search/ajax/?' +\
        'project={0}&'.format(param_dict['project']) +\
        'category={0}&'.format(param_dict['category']) +\
        'dateFrom={0}&'.format(param_dict['dateFrom']) +\
        'dateTo={0}&'.format(param_dict['dateTo']) +\
        'page={0}&'.format(param_dict['page']) +\
        'query={0}&'.format(param_dict['query']) +\
        'material={0}'.format(param_dict['material'])
        
        return url
    
    
    def _get_search_table(self, param_dict: dict,
                          include_text: bool = True) -> pd.DataFrame:
        """
        Возвращает pd.DataFrame со списком статей
        
        include_text: bool
        ### Если True, статьи возвращаются с текстами
        """
        url = self._get_url(param_dict)
        r = rq.get(url)
        search_table = pd.DataFrame(r.json()['items'])
        if include_text and not search_table.empty:
            get_text = lambda x: self._get_article_data(x['fronturl'])
            search_table[['overview', 'text']] = search_table.apply(get_text,
                                                                    axis=1).tolist()
        
        if 'publish_date_t' in search_table.columns:
            search_table.sort_values('publish_date_t', ignore_index=True)
            
        return search_table
    
    
    def _iterable_load_by_page(self, param_dict):
        param_copy = param_dict.copy()
        results = []
        
        result = self._get_search_table(param_copy)
        results.append(result)
        
        while not result.empty:
            param_copy['page'] = str(int(param_copy['page']) + 1)
            result = self._get_search_table(param_copy)
            results.append(result)
            
        results = pd.concat(results, axis=0, ignore_index=True)
        
        return results
    
    
    def _get_article_data(self, url: str):
        """
        Возвращает описание и текст статьи по ссылке
        """
        r = rq.get(url)
        soup = bs(r.text, features="lxml") # features="lxml" чтобы не было warning
        div_overview = soup.find('div', {'class': 'article__text__overview'})
        if div_overview:
            overview = div_overview.text.replace('<br />','\n').strip()
        else:
            overview = None
        p_text = soup.find_all('p')
        if p_text:
            text = ' '.join(map(lambda x:
                                x.text.replace('<br />','\n').strip(),
                                p_text))
        else:
            text = None
        
        return overview, text 
    
    def get_articles(self,
                     param_dict,
                     time_step = 1,
                     save_every = 5,
                     save_excel = True) -> pd.DataFrame:
        """
        Функция для скачивания статей интервалами через каждые time_step дней
        Делает сохранение таблицы через каждые save_every * time_step дней

        param_dict: dict
        ### Параметры запроса 
        ###### project - раздел поиска, например, rbcnews
        ###### category - категория поиска, например, TopRbcRu_economics
        ###### dateFrom - с даты
        ###### dateTo - по дату
        ###### query - поисковой запрос (ключевое слово), например, РБК
        ###### page - смещение поисковой выдачи (с шагом 20)
        
        ###### Deprecated:
        ###### offset - смещение поисковой выдачи
        ###### limit - лимит статей, максимум 100
        """
        param_copy = param_dict.copy()
        time_step = timedelta(days=time_step)
        dateFrom = datetime.strptime(param_copy['dateFrom'], '%d.%m.%Y')
        dateTo = datetime.strptime(param_copy['dateTo'], '%d.%m.%Y')
        if dateFrom > dateTo:
            raise ValueError('dateFrom should be less than dateTo')
        
        out = pd.DataFrame()
        save_counter = 0

        while dateFrom <= dateTo:
            param_copy['dateTo'] = (dateFrom + time_step).strftime("%d.%m.%Y")
            if dateFrom + time_step > dateTo:
                param_copy['dateTo'] = dateTo.strftime("%d.%m.%Y")
            print('Parsing articles from ' + param_copy['dateFrom'] +  ' to ' + param_copy['dateTo'])
            out = pd.concat([out, self._iterable_load_by_page(param_copy)], axis=0, ignore_index=True)
            dateFrom += time_step + timedelta(days=1)
            param_copy['dateFrom'] = dateFrom.strftime("%d.%m.%Y")
            save_counter += 1
            if save_counter == save_every:
                display.clear_output(wait=True)
                out.to_excel("/tmp/checkpoint_table.xlsx")
                print('Checkpoint saved!')
                save_counter = 0
        
        if save_excel:
            out.to_excel("rbc_{}_{}.xlsx".format(
                param_dict['dateFrom'],
                param_dict['dateTo']))
        print('Finish')
        
        return out



use_parser = "РБК"
query = 'РБК'
project = "rbcnews"
category = "TopRbcRu_economics"
material = ""
dateFrom = '2024-04-22'
dateTo = "2024-04-23"
page = 0

if use_parser == "РБК":
    param_dict = {'query'   : query, 
                  'project' : project,
                  'category': category,
                  'dateFrom': datetime.
                  strptime(dateFrom, '%Y-%m-%d').
                  strftime('%d.%m.%Y'),
                  'dateTo'  : datetime.
                  strptime(dateTo, '%Y-%m-%d').
                  strftime('%d.%m.%Y'),
                  'page'   : str(page),
                  'material': material}

print(use_parser, "- param_dict:", param_dict)
assert use_parser == "РБК"
parser = rbc_parser()
tbl = parser._get_search_table(param_dict,
                               include_text = True) # Парсить текст статей
tbl.to_csv('rbk_out.csv', index=False)