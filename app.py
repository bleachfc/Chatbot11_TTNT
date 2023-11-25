from flask import Flask, render_template, request, jsonify
from module.chatbot import response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def index_get():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.get_json().get("message")
        
        # Truyền các đối số cần thiết vào hàm response
        responses = response(text)
        
        message = {"answer": responses}
        return jsonify(message)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
