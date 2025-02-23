from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Server is running! Use /clothes to test."})

@app.route('/clothes', methods=['GET'])
def get_clothes():
    return jsonify({"message": "Clothes route is working!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
