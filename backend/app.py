from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("models/crop_yield_model.pkl")


@app.route("/")
def home():
    return "AI Crop Yield Prediction API Running Successfully!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_df = pd.DataFrame([{
        "Area": data["Area"],
        "Item": data["Item"],
        "Year": data["Year"],
        "average_rain_fall_mm_per_year": data["average_rain_fall_mm_per_year"],
        "pesticides_tonnes": data["pesticides_tonnes"],
        "avg_temp": data["avg_temp"]
    }])

    prediction = model.predict(input_df)[0]

    return jsonify({
        "predicted_yield": round(float(prediction), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)