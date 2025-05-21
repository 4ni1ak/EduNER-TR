from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ner_uygulama import NERPredictor, DemoNERPredictor

app = Flask(__name__)

# Initialize model
try:
    predictor = NERPredictor()
except Exception:
    predictor = DemoNERPredictor()

def convert_to_lowercase(text):
    return text.casefold()

def is_valid_student_id(numara_list):
    return all(numara.isdigit() for numara in numara_list)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    entities = None
    error = None
    example_text = 'murat ostim technical university computer engineering şimşek  220201013'
    project_info = 'This application automatically identifies entities such as university, department, first name, last name, and ID number in Turkish texts. The model is based on Transformer architecture and is fully optimized for Turkish texts.'
    if request.method == 'POST':
        text = request.form.get('text', '')
        text = convert_to_lowercase(text)
        if text.strip():
            entities = predictor.extract_entities(text)
            # ID number validation and correction
            valid_student_ids = []
            unknowns = []
            for student_id in entities["numara"]:
                if student_id.isdigit():
                    valid_student_ids.append(student_id)
                else:
                    new_label = predictor.predict(student_id)
                    if new_label == "B-ISIM":
                        entities["isim"].append(student_id)
                    elif new_label == "B-SOYISIM":
                        entities["soyisim"].append(student_id)
                    else:
                        unknowns.append(student_id)
            entities["numara"] = valid_student_ids
            if unknowns:
                entities.setdefault("unknown", []).extend(unknowns)
            result = entities
    return render_template('index.html', result=result, entities=entities, error=error, example_text=example_text, project_info=project_info)

@app.route('/api/ner', methods=['POST'])
def api_ner():
    data = request.get_json()
    text = data.get('text', '')
    text = convert_to_lowercase(text)
    if not text.strip():
        return jsonify({'error': 'Empty text'}), 400
    entities = predictor.extract_entities(text)
    # ID number validation and correction
    valid_student_ids = []
    unknowns = []
    for student_id in entities["numara"]:
        if student_id.isdigit():
            valid_student_ids.append(student_id)
        else:
            new_label = predictor.predict(student_id)
            if new_label == "B-ISIM":
                entities["isim"].append(student_id)
            elif new_label == "B-SOYISIM":
                entities["soyisim"].append(student_id)
            else:
                unknowns.append(student_id)
    entities["numara"] = valid_student_ids
    if unknowns:
        entities.setdefault("unknown", []).extend(unknowns)
    return jsonify(entities)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 