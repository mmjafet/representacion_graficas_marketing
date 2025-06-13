from flask import Flask, send_file, request, jsonify
from utils.data_processor import (
    load_data, 
    preprocess_data, 
    get_sales_summary,
    group_sales_by_date,
    get_correlation_matrix,
    get_sales_distribution,
    perform_pca_analysis
)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns

app = Flask(__name__)

data_path = os.path.join(os.path.dirname(__file__), 'data', 'sales_data_sample.csv')

try:
    sales_df = load_data(data_path)
    if isinstance(sales_df, str):
        print(f"Error loading data: {sales_df}")
        sales_df = pd.DataFrame({
            'ORDERDATE': pd.date_range(start='2020-01-01', periods=100),
            'SALES': np.random.rand(100) * 1000,
            'QUANTITY': np.random.randint(1, 50, 100),
            'PRICE': np.random.rand(100) * 100,
            'PRODUCTLINE': np.random.choice(['Classic Cars', 'Motorcycles', 'Planes'], 100)
        })
    sales_df = preprocess_data(sales_df)
except Exception as e:
    print(f"Error during data loading/preprocessing: {e}")
    sales_df = pd.DataFrame({
        'ORDERDATE': pd.date_range(start='2020-01-01', periods=100),
        'SALES': np.random.rand(100) * 1000,
        'QUANTITY': np.random.randint(1, 50, 100),
        'PRICE': np.random.rand(100) * 100,
        'PRODUCTLINE': np.random.choice(['Classic Cars', 'Motorcycles', 'Planes'], 100)
    })

def generate_image_response(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

@app.route('/api/correlation_matrix')
def correlation_matrix():
    try:
        corr = sales_df.select_dtypes(include=['number']).corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        return generate_image_response(fig)
    except Exception as e:
        return jsonify({"error": f"Error generating correlation matrix: {str(e)}"}), 500

@app.route('/api/sales_by_date')
def sales_by_date():
    try:
        grouped = group_sales_by_date(sales_df)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(grouped['ORDERDATE'], grouped['SALES'], marker='o')
        ax.set_title('Sales Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        fig.autofmt_xdate()
        return generate_image_response(fig)
    except Exception as e:
        return jsonify({"error": f"Error generating sales over time chart: {str(e)}"}), 500

@app.route('/api/distribution')
def distribution():
    try:
        column = request.args.get('column', 'SALES')
        if column not in sales_df.columns:
            return jsonify({"error": f"Column {column} not found"}), 400
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(sales_df[column], kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
        return generate_image_response(fig)
    except Exception as e:
        return jsonify({"error": f"Error generating distribution: {str(e)}"}), 500

# Puedes mantener estos endpoints JSON si los usas aún en otra interfaz
@app.route('/api/columns')
def get_columns():
    try:
        numeric_columns = sales_df.select_dtypes(include=['number']).columns.tolist()
        return jsonify({"columns": numeric_columns})
    except Exception as e:
        return jsonify({"error": f"Error getting column names: {str(e)}"}), 500

@app.route('/api/pca')
def pca_analysis():
    try:
        n_components = int(request.args.get('components', 3))
        apply_clustering = request.args.get('clustering', 'true').lower() == 'true'
        n_clusters = int(request.args.get('n_clusters', 3))
        pca_data = perform_pca_analysis(
            sales_df, 
            n_components=n_components,
            apply_clustering=apply_clustering,
            n_clusters=n_clusters
        )
        return jsonify(pca_data)
    except Exception as e:
        return jsonify({"error": f"Error generating PCA: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
