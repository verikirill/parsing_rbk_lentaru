#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример использования анализатора ОКВЭД для анализа новостей

Этот скрипт демонстрирует, как использовать OKVEDAnalyzer для анализа
CSV файла с новостями и определения соответствующих кодов ОКВЭД.
"""

from okved_analyzer import OKVEDAnalyzer
import pandas as pd
import logging

def analyze_parsed_news_example():
    """
    Пример анализа новостей, полученных из основного парсера
    """
    # Предполагаем, что у нас есть CSV файл с новостями от основного парсера
    # Формат: id, title, text, date, source, found_companies, sentiment, sentiment_score
    
    input_csv = "parsed_articles_2024-01-01_2024-01-31_банки.csv"  # Замените на ваш файл
    
    try:
        # Создаем анализатор ОКВЭД
        print("Инициализация анализатора ОКВЭД...")
        analyzer = OKVEDAnalyzer(similarity_threshold=0.25)  # Порог схожести 0.25
        
        # Анализируем новости
        print("Начинаем анализ новостей...")
        results = analyzer.analyze_news_csv(input_csv)
        
        # Выводим статистику
        total_news = len(results['news_id'].unique())
        news_with_okved = len(results[results['okved_code'] != '']['news_id'].unique())
        total_okved_matches = len(results[results['okved_code'] != ''])
        
        print(f"\n📊 СТАТИСТИКА АНАЛИЗА:")
        print(f"Всего новостей: {total_news}")
        print(f"Новостей с найденными кодами ОКВЭД: {news_with_okved}")
        print(f"Всего найдено соответствий ОКВЭД: {total_okved_matches}")
        print(f"Среднее количество кодов на новость: {total_okved_matches / news_with_okved:.2f}")
        
        # Показываем топ-5 наиболее часто встречающихся кодов ОКВЭД
        if total_okved_matches > 0:
            top_codes = results[results['okved_code'] != '']['okved_code'].value_counts().head()
            print(f"\n🏆 ТОП-5 НАИБОЛЕЕ ЧАСТЫХ КОДОВ ОКВЭД:")
            for code, count in top_codes.items():
                description = results[results['okved_code'] == code]['okved_description'].iloc[0]
                print(f"{code}: {count} раз - {description[:100]}...")
        
        # Показываем несколько примеров
        print(f"\n📰 ПРИМЕРЫ АНАЛИЗА:")
        sample_results = results[results['okved_code'] != ''].head(3)
        for _, row in sample_results.iterrows():
            print(f"\nНовость: {row['news_title'][:100]}...")
            print(f"ОКВЭД: {row['okved_code']} - {row['okved_description']}")
            print(f"Схожесть: {row['similarity_score']}")
        
    except FileNotFoundError:
        print(f"❌ Файл {input_csv} не найден.")
        print("Сначала запустите основной парсер для создания CSV файла с новостями.")
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")

def test_single_news_analysis():
    """
    Пример анализа отдельной новости
    """
    print("\n🔍 ТЕСТ АНАЛИЗА ОТДЕЛЬНОЙ НОВОСТИ:")
    
    # Пример новости о банках
    news_title = "Сбербанк увеличил процентные ставки по вкладам"
    news_text = """
    Крупнейший российский банк Сбербанк объявил об увеличении процентных ставок 
    по срочным вкладам физических лиц. Новые ставки вступают в силу с 1 февраля. 
    Максимальная ставка по рублевым вкладам составит 8.5% годовых. 
    Банк также расширил линейку депозитных продуктов для корпоративных клиентов.
    """
    
    try:
        analyzer = OKVEDAnalyzer(similarity_threshold=0.2)
        relevant_codes = analyzer.find_relevant_okved_codes(news_text, news_title, top_k=3)
        
        print(f"Заголовок: {news_title}")
        print(f"Найдено кодов ОКВЭД: {len(relevant_codes)}")
        
        for i, code_info in enumerate(relevant_codes, 1):
            print(f"\n{i}. Код ОКВЭД: {code_info['okved_code']}")
            print(f"   Описание: {code_info['okved_description']}")
            print(f"   Схожесть: {code_info['similarity_score']}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def create_sample_csv():
    """
    Создает пример CSV файла для тестирования
    """
    sample_data = [
        {
            'id': '1',
            'title': 'Сбербанк запустил новую кредитную программу',
            'text': 'Крупнейший российский банк Сбербанк объявил о запуске новой кредитной программы для малого и среднего бизнеса. Ставки по кредитам снижены до 12% годовых.',
            'date': '2024-01-15',
            'source': 'RBC.ru',
            'found_companies': "['Сбербанк']",
            'sentiment': 'положительная',
            'sentiment_score': 0.75
        },
        {
            'id': '2', 
            'title': 'Автоваз объявил о сокращении производства',
            'text': 'Крупнейший российский автопроизводитель Автоваз сообщил о временном сокращении объемов производства из-за дефицита комплектующих.',
            'date': '2024-01-16',
            'source': 'RBC.ru',
            'found_companies': "['Автоваз']",
            'sentiment': 'отрицательная',
            'sentiment_score': 0.25
        },
        {
            'id': '3',
            'title': 'Рост цен на нефть поддержал рубль',
            'text': 'Цены на нефть марки Brent выросли до 85 долларов за баррель, что поддержало курс российского рубля. Эксперты прогнозируют дальнейшее укрепление национальной валюты.',
            'date': '2024-01-17',
            'source': 'RBC.ru', 
            'found_companies': "[]",
            'sentiment': 'положительная',
            'sentiment_score': 0.68
        }
    ]
    
    df = pd.DataFrame(sample_data)
    filename = 'sample_news_for_okved_test.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"✅ Создан тестовый файл: {filename}")
    return filename

def main():
    """
    Основная функция для демонстрации возможностей анализатора
    """
    print("🚀 ДЕМОНСТРАЦИЯ АНАЛИЗАТОРА ОКВЭД")
    print("=" * 50)
    
    # Тест анализа отдельной новости
    test_single_news_analysis()
    
    print("\n" + "=" * 50)
    
    # Создаем тестовый файл и анализируем его
    print("\n📝 СОЗДАНИЕ ТЕСТОВОГО ФАЙЛА И АНАЛИЗ:")
    sample_file = create_sample_csv()
    
    try:
        analyzer = OKVEDAnalyzer(similarity_threshold=0.2)
        results = analyzer.analyze_news_csv(sample_file)
        
        print(f"\n📊 Результаты анализа тестового файла:")
        print(f"Обработано новостей: {len(results['news_id'].unique())}")
        print(f"Найдено соответствий: {len(results[results['okved_code'] != ''])}")
        
        # Показываем результаты для каждой новости
        for news_id in results['news_id'].unique():
            news_results = results[results['news_id'] == news_id]
            news_title = news_results['news_title'].iloc[0]
            print(f"\nНовость {news_id}: {news_title[:80]}...")
            
            relevant_codes = news_results[news_results['okved_code'] != '']
            if len(relevant_codes) > 0:
                for _, row in relevant_codes.iterrows():
                    print(f"  - {row['okved_code']}: {row['okved_description'][:60]}... (схожесть: {row['similarity_score']})")
            else:
                print("  - Соответствующие коды ОКВЭД не найдены")
        
    except Exception as e:
        print(f"❌ Ошибка при анализе тестового файла: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Демонстрация завершена!")
    print("\nДля анализа ваших новостей используйте:")
    print("python okved_analyzer.py <путь_к_вашему_csv_файлу>")

if __name__ == "__main__":
    main() 