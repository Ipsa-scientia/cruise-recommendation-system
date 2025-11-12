import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.tagging import assign_nature_weights, CATEGORIES
from sklearn.preprocessing import MinMaxScaler
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2 + 1e-9)

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def analyze_locations(like_locations, dislike_locations, formula_choice, data,
                      duration_min=None, duration_max=None,
                      price_min=None, price_max=None, russian_groups=None):
    try:
        if data is None:
            raise ValueError("Данные о круизах не загружены")

        filtered_data = data.copy()

        filtered_data = data.copy()
        
        # Фильтр по продолжительности
        if duration_min not in [None, '']:
            filtered_data = filtered_data[filtered_data['Продолжительность'] >= int(duration_min)]
        if duration_max not in [None, '']:
            filtered_data = filtered_data[filtered_data['Продолжительность'] <= int(duration_max)]
        
        filtered_data['Цена в евро'] = (
        filtered_data['Цена в евро']
        .astype(str)
        .str.replace(r'[^\d.,]', '', regex=True)
        .str.replace(',', '.', regex=False)
        .str.replace(' ', '', regex=False)
        .astype(float)
        )
        filtered_data = filtered_data.dropna(subset=['Цена в евро'])

        # Фильтр по цене
        if price_min not in [None, '']:
            filtered_data = filtered_data[filtered_data['Цена в евро'] >= float(price_min)]
        if price_max not in [None, '']:
            filtered_data = filtered_data[filtered_data['Цена в евро'] <= float(price_max)]
            
        # Фильтр по русским группам
        if russian_groups == 'yes':
            filtered_data = filtered_data[filtered_data['Русские группы'] == 1]
        elif russian_groups == 'no':
            filtered_data = filtered_data[filtered_data['Русские группы'] == 0]
            
        print("После всех фильтров:", len(filtered_data))

        if len(filtered_data) == 0:
            raise ValueError("Нет круизов, соответствующих выбранным фильтрам")

        filtered_data = filtered_data.reset_index(drop=True)

        if len(filtered_data) == 0:
            raise ValueError("Нет круизов, соответствующих выбранным фильтрам")
        filtered_data = filtered_data.reset_index(drop=True)

        for category in CATEGORIES:
            if category not in filtered_data.columns:
                filtered_data[category] = 0.0
        
        # Заполняем теги на основе описаний
        if 'теги' in filtered_data.columns or 'описание' in filtered_data.columns:
            tag_column = 'теги' if 'теги' in filtered_data.columns else 'описание'
            
            for idx, row in filtered_data.iterrows():
                tags_text = row[tag_column] if pd.notna(row[tag_column]) else ""
                tag_weights = assign_nature_weights(tags_text)
                
                for category, weight in tag_weights.items():
                    filtered_data.at[idx, category] = weight
        
        if len(filtered_data) > 0:
            print({cat: filtered_data.iloc[0][cat] for cat in CATEGORIES})

        # Получаем индексы выбранных круизов с проверкой
        like_indices = []
        for name in like_locations:
            matches = filtered_data[filtered_data['название'] == name]
            if not matches.empty:
                like_indices.append(matches.index[0])
        
        dislike_indices = []
        for name in dislike_locations:
            matches = filtered_data[filtered_data['название'] == name]
            if not matches.empty:
                dislike_indices.append(matches.index[0])
        # Создаем матрицу признаков для всех круизов
        season_features = filtered_data['Сезонность'].values.reshape(-1, 1)
        price_features = filtered_data['Цена в евро'].values.reshape(-1, 1)
        # Для тегов используем все категории
        tag_columns = CATEGORIES
        # Проверяем, что все тег-колонки существуют
        available_tag_columns = [col for col in tag_columns if col in filtered_data.columns]
        if not available_tag_columns:
            print("Предупреждение: Тег-колонки не найдены, создаем пустые...")
            for col in tag_columns:
                if col not in filtered_data.columns:
                    filtered_data[col] = 0.0
            available_tag_columns = tag_columns
        
        tag_features = filtered_data[available_tag_columns].values
        
        print(f"Используемые тег-колонки: {available_tag_columns}")
        print(f"Размер tag_features: {tag_features.shape}")
        
        # Нормализуем все признаки
        season_min, season_max = season_features.min(), season_features.max()
        price_min_val, price_max_val = price_features.min(), price_features.max()
        
        season_norm = (season_features - season_min) / (season_max - season_min) if (season_max - season_min) > 0 else np.zeros_like(season_features)
        price_norm = (price_features - price_min_val) / (price_max_val - price_min_val) if (price_max_val - price_min_val) > 0 else np.zeros_like(price_features)
        
        # Нормализуем теги
        if tag_features.size > 0:
            tag_min = tag_features.min(axis=0)
            tag_max = tag_features.max(axis=0)
            tag_norm = (tag_features - tag_min) / (tag_max - tag_min)
            # Заменяем NaN на 0, если есть деление на ноль
            tag_norm = np.nan_to_num(tag_norm)
        else:
            tag_norm = np.zeros((len(filtered_data), 1))
        
        print(f"После нормализации: season_norm={season_norm.shape}, price_norm={price_norm.shape}, tag_norm={tag_norm.shape}")
        
        # Комбинируем все признаки
        try:
            combined_features = np.hstack([season_norm, price_norm, tag_norm])
            print(f"Комбинированные признаки: {combined_features.shape}")
        except ValueError as e:
            print(f"Ошибка комбинирования: {e}")
            # Если не получается скомбинировать, используем только сезоны и цену
            combined_features = np.hstack([season_norm, price_norm])
            print(f"Используем упрощенные признаки: {combined_features.shape}")
        
        # Вычисляем оценки схожести
        scores = []
        for pos, (idx, row) in enumerate(filtered_data.reset_index(drop=True).iterrows()):
            cruise_name = row['название']
            
            # Пропускаем уже выбранные круизы
            if cruise_name in like_locations or cruise_name in dislike_locations:
                continue
            
            total_score = 0
            
            # Только сезоны
            if formula_choice == '1':
                # Сравниваем с понравившимися круизами
                for like_idx in like_indices:
                    vec1 = season_features[idx]
                    vec2 = season_features[like_idx]
                    similarity = cosine_similarity(vec1, vec2)
                    total_score += similarity
                
                # Учитываем непонравившиеся круизы
                for dislike_idx in dislike_indices:
                    vec1 = season_features[idx]
                    vec2 = season_features[dislike_idx]
                    similarity = cosine_similarity(vec1, vec2)
                    total_score -= similarity
            # Только цена
            elif formula_choice == '2':
                # Сравниваем с понравившимися круизами
                for like_idx in like_indices:
                    vec1 = price_features[idx]
                    vec2 = price_features[like_idx]
                    distance = manhattan_distance(vec1, vec2)
                    total_score -= distance  # Меньшее расстояние = большее сходство
                
                # Учитываем непонравившиеся круизы
                for dislike_idx in dislike_indices:
                    vec1 = price_features[idx]
                    vec2 = price_features[dislike_idx]
                    distance = manhattan_distance(vec1, vec2)
                    total_score += distance  # Большее расстояние = меньшее сходство
            # Только теги
            elif formula_choice == '3':
                # Сравниваем с понравившимися круизами
                for like_idx in like_indices:
                    vec1 = tag_features[idx]
                    vec2 = tag_features[like_idx]
                    distance = euclidean_distance(vec1, vec2)
                    total_score -= distance  # Меньшее расстояние = большее сходство
                
                # Учитываем непонравившиеся круизы
                for dislike_idx in dislike_indices:
                    vec1 = tag_features[idx]
                    vec2 = tag_features[dislike_idx]
                    distance = euclidean_distance(vec1, vec2)
                    total_score += distance  # Большее расстояние = меньшее сходство
            # Комбинированный метод
            elif formula_choice == '4':
                # Веса для каждого типа признаков
                weights = {'season': 0.2, 'price': 0.4, 'tags': 0.4}
                
                # Сравниваем с понравившимися круизами
                for like_idx in like_indices:
                    # Сезоны (косинусное сходство)
                    vec1 = season_features[idx]
                    vec2 = season_features[like_idx]
                    season_sim = cosine_similarity(vec1, vec2)
                    
                    # Цена (манхэттенское расстояние, нормализованное)
                    vec1 = price_features[idx]
                    vec2 = price_features[like_idx]
                    price_dist = manhattan_distance(vec1, vec2)
                    max_price_dist = np.max(np.abs(price_features - price_features.mean()))
                    price_sim = 1 - (price_dist / max_price_dist) if max_price_dist > 0 else 0
                    
                    # Теги (евклидово расстояние, нормализованное)
                    vec1 = tag_features[idx]
                    vec2 = tag_features[like_idx]
                    tag_dist = euclidean_distance(vec1, vec2)
                    max_tag_dist = np.max(np.linalg.norm(tag_features - tag_features.mean(axis=0), axis=1))
                    tag_sim = 1 - (tag_dist / max_tag_dist) if max_tag_dist > 0 else 0
                    
                    # Комбинированная оценка
                    combined_sim = (weights['season'] * season_sim + 
                                  weights['price'] * price_sim + 
                                  weights['tags'] * tag_sim)
                    total_score += combined_sim
                
                # Учитываем непонравившиеся круизы
                for dislike_idx in dislike_indices:
                    # Сезоны (косинусное сходство)
                    vec1 = season_features[idx]
                    vec2 = season_features[dislike_idx]
                    season_sim = cosine_similarity(vec1, vec2)
                    
                    # Цена (манхэттенское расстояние, нормализованное)
                    vec1 = price_features[idx]
                    vec2 = price_features[dislike_idx]
                    price_dist = manhattan_distance(vec1, vec2)
                    max_price_dist = np.max(np.abs(price_features - price_features.mean()))
                    price_sim = 1 - (price_dist / max_price_dist) if max_price_dist > 0 else 0
                    
                    # Теги (евклидово расстояние, нормализованное)
                    vec1 = tag_features[idx]
                    vec2 = tag_features[dislike_idx]
                    tag_dist = euclidean_distance(vec1, vec2)
                    max_tag_dist = np.max(np.linalg.norm(tag_features - tag_features.mean(axis=0), axis=1))
                    tag_sim = 1 - (tag_dist / max_tag_dist) if max_tag_dist > 0 else 0
                    
                    # Комбинированная оценка
                    combined_sim = (weights['season'] * season_sim + 
                                  weights['price'] * price_sim + 
                                  weights['tags'] * tag_sim)
                    total_score -= combined_sim
            
            # Нормализуем оценку
            norm_factor = len(like_indices) + len(dislike_indices)
            if norm_factor > 0:
                total_score /= norm_factor
            
            scores.append((cruise_name, total_score))
        
        recommendations = sorted(scores, key=lambda x: x[1], reverse=True)

        return recommendations[:10], None

    except Exception as e:
        error_msg = f"Ошибка при анализе: {str(e)}"
        return None, error_msg
