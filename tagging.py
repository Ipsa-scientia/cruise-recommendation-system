CATEGORIES = ["неживая природа", "живая природа", "активный отдых", "музыкальные", "гастрономия"]

def assign_nature_weights(tags):
    if not isinstance(tags, str):
        return {category: 0.0 for category in CATEGORIES}

    tags = tags.lower()
    weights = {category: 0.0 for category in CATEGORIES}

    keyword_mappings = {
        "неживая природа": ["полюс", "ночь", "затмение", "антарктида", "амазонка"],
        "живая природа": ["медведь", "пингвины", "рифы", "медузы"],
        "активный отдых": ["дайвинг", "альпинизм"],
        "музыкальные": ["фестиваль", "опера", "концерты"],
        "гастрономия": ["шеф", "винодельни"]
    }

    for category, keywords in keyword_mappings.items():
        for word in keywords:
            if word in tags:
                weights[category] += 1.0

    total = sum(weights.values())
    if total > 0:
        return {k: round(v / total, 3) for k, v in weights.items()}
    else:
        return {category: 0.0 for category in CATEGORIES}
