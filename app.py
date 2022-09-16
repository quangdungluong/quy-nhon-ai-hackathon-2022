from flask import Flask, request, jsonify, render_template
import app_config
import settings
from solver import ClassifyReviewSolver
import pandas as pd

review_solver = ClassifyReviewSolver(app_config)

app = Flask(__name__)

@app.get("/")
def root():
    return render_template('index.html')

@app.post("/")
def main():
    RATING_ASPECTS = ["Dịch vụ giải trí", "Dịch vụ lưu trứ", "Dịch vụ nhà hàng", "Dịch vụ ăn uống", "Dịch vụ di chuyển", "Dịch vụ mua sắm"]
    if request.method == "POST":
        review_sentence = request.form["review"]
        predict_results = review_solver.solve(review_sentence)

        for i in range(len(predict_results)):
            if predict_results[i] > 0:
                predict_results[i] = str(predict_results[i]) + "⭐"
        
        data = dict(zip(RATING_ASPECTS, predict_results))
        data = pd.DataFrame([data])
        return render_template("index.html", data=data.to_html(classes='table table-stripped mycustombtn',index=False,justify="center"), review_sentence=review_sentence)

if __name__ == '__main__':
    app.run(host=settings.HOST, port=settings.PORT)