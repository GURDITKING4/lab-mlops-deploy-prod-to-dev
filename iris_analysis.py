import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Iris Analysis", layout="wide")

# Title
st.title("🌸 Iris Flower Analysis")
st.markdown("**Explore and analyze the famous Iris dataset**")

# Load data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    return df

df = load_data()

# Sidebar
st.sidebar.header("Dataset Info")
st.sidebar.write(f"Total samples: {len(df)}")
st.sidebar.write(f"Features: {len(df.columns) - 1}")
st.sidebar.write(f"Species: {df['species'].nunique()}")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.subheader("Species Distribution")
    species_count = df['species'].value_counts()
    fig = px.bar(x=species_count.index, y=species_count.values, 
                 labels={'x': 'Species', 'y': 'Count'},
                 color=species_count.index)
    st.plotly_chart(fig, use_container_width=True)

# Feature analysis
st.subheader("Feature Analysis")
feature = st.selectbox("Select a feature:", df.columns[:-1])
fig = px.histogram(df, x=feature, color='species', nbins=20)
st.plotly_chart(fig, use_container_width=True)

# PCA visualization
st.subheader("PCA Visualization")
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df.iloc[:, :-1])
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['species'] = df['species'].values

fig = px.scatter(pca_df, x='PC1', y='PC2', color='species',
                title=f'PCA: {pca.explained_variance_ratio_.sum():.2%} Variance Explained')
st.plotly_chart(fig, use_container_width=True)

# Statistics
st.subheader("Statistics by Species")
st.dataframe(df.groupby('species').describe().round(2), use_container_width=True)
