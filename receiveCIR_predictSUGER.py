from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('wm_classification_model.h5')

# 텍스트 파일을 받아서 처리하는 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # 텍스트 파일 파싱 (예시: 파일을 텍스트로 변환)
        text = file.read().decode('utf-8')

        # 텍스트 파일을 모델이 이해할 수 있는 형태로 변환 (전처리)
        processed_data = preprocess(text)

        # 모델을 통해 예측 수행
        prediction = model.predict(np.array([processed_data]))

        index = np.argmax(prediction, axis=-1)

        # 결과 반환
        return jsonify({'prediction': index})

def preprocess(text):
    # 데이터를 "/"로 분리하여 각 수집 데이터를 나눔
    records = text.strip().split('/')
    
    # 각 레코드를 공백으로 분리하여 숫자 리스트로 변환
    data = [list(map(float, record.split())) for record in records if record.strip()]
    
   # [430][80]의 numpy 배열로 변환하여 리턴
    array_data = np.array(data)
    return array_data.reshape(430, 80)  # 430개의 데이터 묶음, 각 묶음에 80개의 숫자가 들어가도록 reshape


if __name__ == '__main__':
    app.run(debug=True)