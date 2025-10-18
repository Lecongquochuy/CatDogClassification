from flask import Flask, request, jsonify
from model import load_model, predict_image
from PIL import Image
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

model, transform = load_model('best_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("⚠️ Không có file 'image' trong request.files")
        return jsonify({'Error': 'no image upload'}), 400

    file = request.files['image']
    img_bytes = file.read()

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        print("⚠️ Ảnh không hợp lệ:", e)
        return jsonify({'error': 'invalid image', 'detail': str(e)}), 400

    label, probs = predict_image(model, transform, img)
    print(f"✅ Predict: {label} | probs = {probs}")
    return jsonify({'label': label, 'probabilities': probs})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)