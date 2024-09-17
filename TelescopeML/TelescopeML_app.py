import streamlit as st
import pandas as pd
import bz2
import io
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from unsupervised_ml import TrainMlUnsupervised  # Import the custom module

# Custom styles for the app
st.markdown(
    """
    <style>
    .main {
        background-color: #FFFFFF; 
        color: #2F4F4F; 
    }
    .stButton>button {
        background-color: #007BFF; 
        color: black; 
        border-radius: 10px; 
        border: none; 
    }
    .stButton>button:hover {
        background-color: #0056b3; 
        color: black; 
    }
    .stSlider>div { 
        color: #2F4F4F; 
    }
    .stSelectbox>div { 
        color: #829595; 
    }
    .stSelectbox select { 
        background-color: #555555; 
        color: white; 
        border-radius: 5px; 
        padding: 5px; 
        border: 1px solid #555555; 
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #000000;'>Enhancing TelescopeML With Unsupervised Learning</h1>", unsafe_allow_html=True)

# File uploader to select data file
uploaded_file = st.file_uploader("Choose a CSV file or bz2 compressed file", type=["csv", "bz2"])

def load_data(file):
    try:
        if file.name.endswith('.bz2'):
            with bz2.BZ2File(file, 'rb') as bz2_file:
                data = bz2_file.read()
                df = pd.read_csv(io.BytesIO(data))
        else:
            # Read CSV file directly
            df = pd.read_csv(file)
        
        # Instantiating the class from the unsupervised_ml module
        ml_model = TrainMlUnsupervised()
        
        # Assuming TrainMlUnsupervised has a method to preprocess data
        df = ml_model.load_and_preprocess_data(df)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        st.write(f'**Loaded Data (showing first 5 rows):**')
        st.write(df.head())
        st.write(f'**Initial Data Shape:** {df.shape}')

        feature_values = df.values
        feature_names = df.columns.tolist()

        # Standardizing features manually
        scaler = StandardScaler()
        feature_values_standardized = scaler.fit_transform(feature_values)

        feature_df = pd.DataFrame(feature_values_standardized, columns=feature_names)

        # User selects axes and method
        st.write("### Select X and Y Axes for Visualization")
        x_axis = st.selectbox('Select X-axis:', options=feature_names)
        y_axis = st.selectbox('Select Y-axis:', options=feature_names)

        st.write("### Clustering and Dimensionality Reduction Methods")
        method = st.selectbox('Choose Method', ['Choose Method', 'K-Means', 'PCA', 'DBSCAN'], key='method_select')

        if method == 'K-Means':
            n_clusters = st.slider('Number of Clusters', min_value=2, max_value=10, value=3)
        elif method == 'PCA':
            n_components = st.slider('Number of Components', min_value=2, max_value=min(df.shape[1], 3), value=2)
        elif method == 'DBSCAN':
            eps = st.slider('Epsilon (eps)', min_value=0.1, max_value=10.0, value=0.5, step=0.1)
            min_samples = st.slider('Minimum Samples', min_value=1, max_value=20, value=5)

        if method != 'Choose Method' and st.button('Run'):
            try:
                if method == 'K-Means':
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(feature_values_standardized)
                    feature_df['Cluster Label'] = cluster_labels

                    # Debugging: Display some cluster labels
                    st.write("Cluster Labels:")
                    st.write(feature_df[['Cluster Label']].head())

                    # Compute silhouette score
                    silhouette_avg = silhouette_score(feature_values_standardized, cluster_labels)
                    st.write(f'Silhouette Score for K-Means: {silhouette_avg:.4f}')

                    fig = px.scatter(
                        feature_df,
                        x=x_axis,
                        y=y_axis,
                        color='Cluster Label',
                        title='K-Means Clustering Visualization',
                        labels={'Cluster Label': 'Cluster'},
                        color_continuous_scale=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig)

                elif method == 'PCA':
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(feature_values_standardized)

                    st.write(f'**Shape After PCA:** {X_pca.shape}')
                    st.write('PCA Components:')
                    st.write(pd.DataFrame(X_pca))

                    explained_variance = pca.explained_variance_ratio_
                    fig = px.bar(
                        x=range(1, len(explained_variance) + 1),
                        y=explained_variance,
                        labels={'x': 'Principal Component', 'y': 'Variance Explained'},
                        title='Explained Variance by PCA Components'
                    )
                    st.plotly_chart(fig)

                    if n_components == 2:
                        fig = px.scatter(
                            x=X_pca[:, 0],
                            y=X_pca[:, 1],
                            title='PCA Components Visualization',
                            labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'}
                        )
                        st.plotly_chart(fig)
                    elif n_components == 3:
                        fig = px.scatter_3d(
                            x=X_pca[:, 0],
                            y=X_pca[:, 1],
                            z=X_pca[:, 2],
                            title='3D PCA Components Visualization',
                            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'}
                        )
                        st.plotly_chart(fig)

                elif method == 'DBSCAN':
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = dbscan.fit_predict(feature_values_standardized)
                    feature_df['Cluster Label'] = cluster_labels

                    # Debugging: Display some cluster labels
                    st.write("Cluster Labels:")
                    st.write(feature_df[['Cluster Label']].head())

                    # Compute silhouette score only for valid clusters
                    if len(set(cluster_labels)) > 1:  # Ensure there are more than one cluster
                        silhouette_avg = silhouette_score(feature_values_standardized, cluster_labels)
                        st.write(f'Silhouette Score for DBSCAN: {silhouette_avg:.4f}')
                    else:
                        st.write("Silhouette Score cannot be computed for a single cluster.")

                    fig = px.scatter(
                        feature_df,
                        x=x_axis,
                        y=y_axis,
                        color='Cluster Label',
                        title='DBSCAN Clustering Visualization',
                        labels={'Cluster Label': 'Cluster'},
                        color_continuous_scale=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"An error occurred during clustering: {e}")

else:
    st.write("Please upload a file to begin.")
