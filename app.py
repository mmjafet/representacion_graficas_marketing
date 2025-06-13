from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

DATA_FILE = 'dataset.csv'
df_global = None

# --- Funciones de procesamiento ---

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        return str(e)

def preprocess_data(df):
    if 'ORDERDATE' in df.columns:
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
        df = df.dropna(subset=['ORDERDATE'])
        df = df.sort_values('ORDERDATE')
    return df

def get_sales_summary(df):
    return {
        "total_sales": float(df['SALES'].sum()),
        "average_sales": float(df['SALES'].mean()),
        "max_sales": float(df['SALES'].max()),
        "min_sales": float(df['SALES'].min()),
        "total_orders": int(df.shape[0])
    }

def group_sales_by_date(df):
    grouped = df.groupby('ORDERDATE', as_index=False)['SALES'].sum()
    grouped['ORDERDATE'] = grouped['ORDERDATE'].dt.strftime('%Y-%m-%d')
    return grouped.to_dict(orient='records')

def get_correlation_matrix(df):
    corr = df.select_dtypes(include=['number']).corr()
    return corr.to_dict()

def get_sales_distribution(df, column='SALES'):
    if column not in df.columns:
        return None
    return {
        "mean": float(df[column].mean()),
        "std": float(df[column].std()),
        "min": float(df[column].min()),
        "max": float(df[column].max()),
        "25%": float(df[column].quantile(0.25)),
        "50%": float(df[column].median()),
        "75%": float(df[column].quantile(0.75))
    }

def perform_pca_analysis(df, n_components=3, apply_clustering=True, n_clusters=3):
    df_num = df.select_dtypes(include=['number']).dropna()
    
    if df_num.shape[0] == 0 or df_num.shape[1] < n_components:
        return {"error": "No hay suficientes datos numéricos para realizar PCA"}

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_scaled)

    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])

    result = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca_df.to_dict(orient='records')
    }

    if apply_clustering:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(pca_df)
        pca_df['cluster'] = clusters
        result["components"] = pca_df.to_dict(orient='records')

    return result

# --- Rutas API ---

def check_data_loaded():
    global df_global
    if df_global is None:
        return False, jsonify({"error": "No data loaded. Upload a CSV file via POST /upload or place dataset.csv in the server."}), 400
    return True, None, None

@app.route('/')
def home():
    return jsonify({"message": "API activa. Sube tu dataset con POST /upload y luego usa los endpoints GET."})

@app.route('/upload', methods=['POST'])
def upload():
    global df_global
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        df = pd.read_csv(file)
        df = preprocess_data(df)
        df_global = df
        return jsonify({"message": f"Archivo cargado con éxito con {df.shape[0]} registros."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summary')
def summary():
    ok, resp, code = check_data_loaded()
    if not ok:
        return resp, code
    return jsonify(get_sales_summary(df_global))

@app.route('/grouped')
def grouped():
    ok, resp, code = check_data_loaded()
    if not ok:
        return resp, code
    return jsonify(group_sales_by_date(df_global))

@app.route('/correlation')
def correlation():
    ok, resp, code = check_data_loaded()
    if not ok:
        return resp, code
    return jsonify(get_correlation_matrix(df_global))

@app.route('/distribution')
def distribution():
    ok, resp, code = check_data_loaded()
    if not ok:
        return resp, code
    dist = get_sales_distribution(df_global)
    if dist is None:
        return jsonify({"error": "Columna SALES no encontrada"}), 400
    return jsonify(dist)

@app.route('/pca')
def pca():
    ok, resp, code = check_data_loaded()
    if not ok:
        return resp, code
    result = perform_pca_analysis(df_global)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == '__main__':
    # Si dataset.csv existe en el directorio, se carga automáticamente
    if os.path.exists(DATA_FILE):
        df_global = load_data(DATA_FILE)
        if isinstance(df_global, pd.DataFrame):
            df_global = preprocess_data(df_global)
        else:
            print(f"Error cargando {DATA_FILE}: {df_global}")

    app.run(host='0.0.0.0', port=8000, debug=True)
