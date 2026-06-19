from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Crop Yield Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    rainfall = float(data["rainfall"])
    temperature = float(data["temperature"])

    prediction = round((rainfall * 0.002) + (temperature * 0.1), 2)

    return jsonify({"predicted_yield": prediction})

if __name__ == "__main__":
    app.run(debug=True)