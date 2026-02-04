from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load saved model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":

        # Get input values from form
        road_type = request.form["road_type"]
        lighting = request.form["lighting"]
        weather = request.form["weather"]
        time_of_day = request.form["time_of_day"]

        speed = float(request.form["speed"])
        traffic_density = float(request.form["traffic_density"])

        # Create DataFrame for prediction
        input_data = pd.DataFrame([{
            "road_type": road_type,
            "lighting": lighting,
            "weather": weather,
            "time_of_day": time_of_day,
            "speed": speed,
            "traffic_density": traffic_density
        }])

        # Predict accident risk
        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
