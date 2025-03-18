# # # from flask import Flask, request, jsonify, render_template
# # # import pickle
# # # import numpy as np

# # # app = Flask(__name__)

# # # # Load trained model
# # # with open("model.pkl", "rb") as f:
# # #     model = pickle.load(f)

# # # @app.route("/")
# # # def home():
# # #     return render_template("index.html")  # Load frontend

# # # @app.route("/predict", methods=["POST"])
# # # def predict():
# # #     data = request.json
# # #     features = np.array([
# # #         data["temperature"], data["humidity"], data["wind_speed"], data["cloud_cover"]
# # #     ]).reshape(1, -1)

# # #     prediction = model.predict(features)
    
# # #     return jsonify({"predicted_power_kw": round(prediction[0], 2)})

# # # if __name__ == "__main__":
# # #     app.run(debug=True)


# # from flask import Flask, render_template, request, jsonify
# # import pickle
# # import numpy as np

# # app = Flask(__name__)

# # # Load the trained model
# # with open("model.pkl", "rb") as file:
# #     model = pickle.load(file)

# # @app.route("/")
# # def home():
# #     return render_template("index.html")  # Load the frontend

# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         # Get user input from form
# #         data = [float(x) for x in request.form.values()]  # Convert input to float
# #         features = np.array([data])  # Convert to numpy array
        
# #         # Make prediction
# #         prediction = model.predict(features)[0]  # Get the first prediction
        
# #         return jsonify({"prediction": prediction})  # Return JSON response
# #     except Exception as e:
# #         return jsonify({"error": str(e)})

# # if __name__ == "__main__":
# #     app.run(debug=True)

# from flask import Flask, request, jsonify
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open("model.pkl", "rb"))

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get JSON data from frontend
#         data = request.json  
#         print("Received Data:", data)  # Debugging line to check received input

#         # Extract features from JSON
#         feature1 = float(data['feature1'])
#         feature2 = float(data['feature2'])
#         feature3 = float(data['feature3'])
        
#         # Make prediction
#         prediction = model.predict([[feature1, feature2, feature3]])

#         # Return prediction as JSON
#         return jsonify({"prediction": prediction[0]})

#     except Exception as e:
#         print("Error:", str(e))
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the frontend
        data = request.get_json()
        feature1 = float(data['feature1'])
        feature2 = float(data['feature2'])
        feature3 = float(data['feature3'])

        # Prepare input for the model
        features = np.array([[feature1, feature2, feature3]])

        # Make a prediction
        prediction = model.predict(features)[0]

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({"message": "Prediction route is working!"})
