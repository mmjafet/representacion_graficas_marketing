from flask import Flask, request, jsonify
import pandas as pd
from funciones import (
    load_data,
    preprocess_data,
    get_sales_summary,
    group_sales_by_date,
    get_correlation_matrix,
    get_sales_distribution,
    perform_pca_analysis
)

app = Flask(__name__)

@app.route("/")
def index():
    return "API de análisis de ventas activa ✅"

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No se envió un archivo"}), 400

    file = request.files['file']
    
    try:
        df = pd.read_csv(file)
        df = preprocess_data(df)

        summary = get_sales_summary(df)
        grouped = group_sales_by_date(df).to_dict(orient='records')
        correlation = get_correlation_matrix(df).to_dict()
        distribution = get_sales_distribution(df)
        pca = perform_pca_analysis(df)

        return jsonify({
            "summary": summary,
            "sales_grouped_by_date": grouped,
            "correlation_matrix": correlation,
            "sales_distribution": distribution,
            "pca": pca
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
