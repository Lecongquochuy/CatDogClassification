from flask import Flask, request, jsonify
from model import get_model, predict_image
from PIL import Image
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app, origins=["https://cat-dog-classification-beryl.vercel.app"], supports_credentials=True)

MODEL_PATHS = {
    "cnn": "weights/best_model.pth",
    "resnet": "weights/resnet_weight.pth",
    "mobilenet": "weights/mobilenet_weight.pth",
    "efficientnet": "weights/efficientnet_weight.pth",
}

# model, transform = load_model('weights/best_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    model_name = request.form.get('model', 'cnn')
    weight_path = MODEL_PATHS.get(model_name, MODEL_PATHS['cnn'])

    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    # Load model + transform
    try:
        model, transform = get_model(model_name, weight_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    image = Image.open(file).convert('RGB')
    label, probs = predict_image(model, transform, image)

    return jsonify({'label': label, 'probabilities': probs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)