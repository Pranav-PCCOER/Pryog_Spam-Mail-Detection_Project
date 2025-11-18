from flask import Flask, render_template, request
from model import predict

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    message = ''
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        if message:
            prediction = predict(message)
    return render_template('home.html', prediction=prediction, message=message)


if __name__ == '__main__':
    app.run(debug=True)
