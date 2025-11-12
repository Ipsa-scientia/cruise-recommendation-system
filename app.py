from flask import Flask, render_template, request, session
from utils.data_loader import load_cruises_data
from utils.similarity import analyze_locations
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CSV_FILE_PATH = Config.CSV_FILE_PATH

@app.route('/', methods=['GET', 'POST'])
def index():
    from utils.data_loader import CATEGORIES

    session.setdefault('likes', [])
    session.setdefault('dislikes', [])

    data = load_cruises_data(CSV_FILE_PATH)
    locations = data['название'].tolist() if data is not None else []

    if request.method == 'POST':
        like_locations = request.form.getlist('like_locations')
        dislike_locations = request.form.getlist('dislike_locations')
        formula_choice = request.form.get('formula', '1')
        duration_min = request.form.get('duration_min')
        duration_max = request.form.get('duration_max')
        price_min = request.form.get('price_min')
        price_max = request.form.get('price_max')
        russian_groups = request.form.get('russian_groups')

        result = analyze_locations(
            like_locations=like_locations,
            dislike_locations=dislike_locations,
            formula_choice=formula_choice,
            data=data,
            duration_min=duration_min,
            duration_max=duration_max,
            price_min=price_min,
            price_max=price_max,
            russian_groups=russian_groups
        )
        
        if result is None:
            error = "Функция analyze_locations вернула None"
            return render_template('index.html', 
                                locations=locations,
                                error=error)
        
        recommendations, error = result
        
        if error:
            return render_template('index.html', 
                                locations=locations,
                                error=error)
        
        if recommendations is None:
            return render_template('index.html',
                                locations=locations,
                                error="Рекомендации не были сгенерированы")
        
        return render_template('index.html',
                            locations=locations,
                            recommendations=recommendations,
                            like_locations=like_locations,
                            dislike_locations=dislike_locations)
    
    return render_template('index.html', locations=locations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)