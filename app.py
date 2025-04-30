from flask import Flask, render_template, jsonify, request
from utils.data_processor import (
    load_data, 
    preprocess_data, 
    get_sales_summary,
    group_sales_by_date,
    get_correlation_matrix,
    get_sales_distribution,
    perform_pca_analysis  # Añadir esta importación
)
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# Configure the data file path
data_path = os.path.join(os.path.dirname(__file__), 'data', 'sales_data_sample.csv')

# Load and preprocess data at startup
try:
    sales_df = load_data(data_path)
    if isinstance(sales_df, str):  # Error returned as string
        print(f"Error loading data: {sales_df}")
        # Create a simple dummy dataframe for testing when file can't be loaded
        sales_df = pd.DataFrame({
            'ORDERDATE': pd.date_range(start='2020-01-01', periods=100),
            'SALES': np.random.rand(100) * 1000,
            'QUANTITY': np.random.randint(1, 50, 100),
            'PRICE': np.random.rand(100) * 100,
            'PRODUCTLINE': np.random.choice(['Classic Cars', 'Motorcycles', 'Planes'], 100)
        })
        print("Created dummy data for testing")
    
    sales_df = preprocess_data(sales_df)
    print("Data loaded and preprocessed successfully")
except Exception as e:
    print(f"Error during data loading/preprocessing: {e}")
    # Create dummy data even on exception
    sales_df = pd.DataFrame({
        'ORDERDATE': pd.date_range(start='2020-01-01', periods=100),
        'SALES': np.random.rand(100) * 1000,
        'QUANTITY': np.random.randint(1, 50, 100),
        'PRICE': np.random.rand(100) * 100,
        'PRODUCTLINE': np.random.choice(['Classic Cars', 'Motorcycles', 'Planes'], 100)
    })
    print("Created dummy data after error")

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/api/correlation_matrix')
def correlation_matrix():
    """API endpoint to get correlation matrix data."""
    try:
        corr_data = get_correlation_matrix(sales_df)
        return jsonify(corr_data)
    except Exception as e:
        return jsonify({"error": f"Error generating correlation matrix: {str(e)}"}), 500

@app.route('/api/sales_by_date')
def sales_by_date():
    """API endpoint to get sales data by date."""
    try:
        grouped_data = group_sales_by_date(sales_df)
        # Convert to dictionary for JSON response
        result = {
            'dates': grouped_data['ORDERDATE'].dt.strftime('%Y-%m-%d').tolist(),
            'sales': grouped_data['SALES'].tolist() if 'SALES' in grouped_data.columns else []
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error generating sales by date: {str(e)}"}), 500

@app.route('/api/distribution')
def distribution():
    """API endpoint to get distribution data for a specific column."""
    try:
        column = request.args.get('column', 'SALES')
        if column not in sales_df.columns:
            return jsonify({"error": f"Column {column} not found"}), 400
        
        data = get_sales_distribution(sales_df, column)
        return jsonify({"column": column, "data": data})
    except Exception as e:
        return jsonify({"error": f"Error generating distribution data: {str(e)}"}), 500

@app.route('/api/columns')
def get_columns():
    """API endpoint to get numeric column names for dropdowns."""
    try:
        numeric_columns = sales_df.select_dtypes(include=['number']).columns.tolist()
        return jsonify({"columns": numeric_columns})
    except Exception as e:
        return jsonify({"error": f"Error getting column names: {str(e)}"}), 500

@app.route('/api/scatter')
def scatter_plot():
    """API endpoint to get data for scatter plot."""
    try:
        x_column = request.args.get('x', 'QUANTITY')
        y_column = request.args.get('y', 'SALES')
        
        if x_column not in sales_df.columns:
            return jsonify({"error": f"X-axis column {x_column} not found"}), 400
        if y_column not in sales_df.columns:
            return jsonify({"error": f"Y-axis column {y_column} not found"}), 400
        
        # Get values for scatter plot
        data = {
            'x': sales_df[x_column].tolist(),
            'y': sales_df[y_column].tolist(),
            'x_label': x_column,
            'y_label': y_column
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Error generating scatter plot data: {str(e)}"}), 500

@app.route('/api/pca')
def pca_analysis():
    """API endpoint para obtener datos de análisis PCA en 3D con clustering opcional."""
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
        return jsonify({"error": f"Error generando análisis PCA: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)