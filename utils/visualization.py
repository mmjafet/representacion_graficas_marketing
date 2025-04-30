from flask import jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('data/sales_data_sample.csv', encoding='unicode_escape')
    return df

def generate_heatmap():
    df = load_data()
    plt.figure(figsize=(20, 20))
    numeric_columns = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cbar=False)
    plt.savefig('static/images/heatmap.png')
    plt.close()

def generate_scatter_matrix():
    df = load_data()
    fig = px.scatter_matrix(df, dimensions=df.columns[:8], color='MONTH_ID')
    fig.write_image('static/images/scatter_matrix.png')

def generate_distribution_plots():
    df = load_data()
    for col in df.columns[:8]:
        if col != 'ORDERLINENUMBER' and pd.api.types.is_numeric_dtype(df[col]):
            data = df[col].dropna().astype(float)
            fig = px.histogram(data, x=data)
            fig.write_image(f'static/images/distplot_{col}.png')

def generate_pca_plot():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = load_data()
    numeric_features = df.select_dtypes(include=['number'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_features)

    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=principal_components, columns=['pca1', 'pca2', 'pca3'])
    fig = px.scatter_3d(pca_df, x='pca1', y='pca2', z='pca3')
    fig.write_image('static/images/pca_plot.png')