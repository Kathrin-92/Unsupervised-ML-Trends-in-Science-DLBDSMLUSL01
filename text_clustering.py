"""
This is the second step of the project.
In this part of the code, four main steps are implemented:
(1) TF-IDF vectorization technique:
    Vectorization technique used on 'abstract' column to represent abstracts as numerical features
(2) Dimensionality Reduction with PCA:
    To reduce the dimensionality of the TF-IDF vectors while preserving the main traits of the dataset
(3) Perform K-Means Clustering on PCA:
    Documents are clustered
(4) Calculate Top Words for each Cluster
"""

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# Standard Library Imports
import os
from itertools import product

# Third-Party Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster.elbow import kelbow_visualizer

# ----------------------------------------------------------------------------------------------------------------------
# LOADING THE DATA
# ----------------------------------------------------------------------------------------------------------------------

# Load cleaned data subset
BASE_PATH = '/YourFilePathHere/'
FILE_NAME = 'data_subset_cleaned.csv'
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)
df = pd.read_csv(FILE_PATH, delimiter=',', encoding='utf-8')
df_abstracts = df['abstract_lemmatized']


# ----------------------------------------------------------------------------------------------------------------------
# TOPIC MODELLING
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------
# PERFORM TF-IDF AND PCA
# ----------------------------------------------------------

# Define tf-idf vectorizer
def tfidf_vectorizer(abstract, max_df, min_df):
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    tfidf_vectors = vectorizer.fit_transform(
        abstract
    )  # TF-IDF = numerical representation of text documents
    tfidf_feature_names = (
        vectorizer.get_feature_names_out()
    )  # feature names represent the vocabs used in whole corpus
    return tfidf_vectors, tfidf_feature_names

# Try different max_ and min_df for tf-idf vectorizer to get the optimal combination
tfidf_data = []

for max_df, min_df in product([0.3, 0.4, 0.5, 0.7], [0.05, 0.1, 0.15]):
    tfidf_vectors, tfidf_feature_names = tfidf_vectorizer(df_abstracts, max_df, min_df)
    num_vectors = tfidf_vectors.shape[0]

    tfidf_data.append(
        {
            "max_df": max_df,
            "min_df": min_df,
            "tfidf_vectors": tfidf_vectors,
            "tfidf_feature_names": tfidf_feature_names,
        }
    )

# find out how many n_components to use for PCA by analyzing explained variance ratios
# try for each tf-idf vectorizer combination
for tfidf_i in tfidf_data:
    tfidf_vectors = tfidf_i["tfidf_vectors"]
    tfidf_feature_names = tfidf_i["tfidf_feature_names"]
    max_df = tfidf_i["max_df"]
    min_df = tfidf_i["min_df"]
    num_vectors = tfidf_vectors.shape[0]
    #print("max_df: {}, min_df: {}, num_vectors: {}".format(max_df, min_df, num_vectors))

    # compute and print explained variance ratios and calculate cumulative explained variance, plot in line graph
    # goal is to understand how much of the overall variance is explained by each principal component
    pca = PCA(n_components=10, random_state=42)
    pca_vectors_reduced = pca.fit(tfidf_vectors.toarray())

    # Calculate explained variance ratios
    # how much of the total variance is explained by each of the principal components
    explained_var_ratio = pca.explained_variance_ratio_
    #print(explained_var_ratio)

    # Calculate cumulative explained variance
    cumulative_explained_var = np.cumsum(explained_var_ratio)
    #print(cumulative_explained_var)

    # Plot the explained variance ratios
    plt.plot(cumulative_explained_var)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance vs. Number of Components")
    plt.show()

# >>> result: best combination for tried out values <<<
# based on quality of extracted keywords for each cluster: max_df = 0.4, min_df = 0.05, n_components = 2
# based on cumulative explained variance: max_df = 0.7, min_df = 0.15, n_components = 2
tfidf_vectors_final, tfidf_feature_names_final = tfidf_vectorizer(df_abstracts, max_df=0.4, min_df=0.05)


# Define PCA vectorizer with above tested number of n_components
def pca_vectorizer(tfidf_vectors):
        pca = PCA(n_components=2, random_state=42)
        pca_vectors_reduced = pca.fit_transform(
            tfidf_vectors.toarray()
        )  # perform PCA dimensionality reduction
        return pca_vectors_reduced


pca_vectors_reduced = pca_vectorizer(tfidf_vectors_final)

# Create a new column in the DataFrame
df["pca_vector_1"] = pca_vectors_reduced[:, 0]  # Select the first column of the 2D array
df["pca_vector_2"] = pca_vectors_reduced[:, 1]  # Select the second column of the 2D array


# ----------------------------------------------------------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------
# CLUSTER THE DATA WITH THE HELP OF K-MEANS
# ----------------------------------------------------------

# find out optimal number of clusters using the elbow method
kelbow_visualizer(
    KMeans(random_state=42),
    pca_vectors_reduced,
    k=(2, 11),
    locate_elbow=True,
    timings=False,
)

optimal_k = 4 # result of kelbow-visualizier for 1969 processed records in category "Quantitative Biology"


# Perform kmeans clustering on the basis of the reduced PCA vectors
def kmeans_clustering(pca_vectors_reduced, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, init="k-means++")
    kmeans.fit(pca_vectors_reduced)
    cluster_labels = kmeans.fit_predict(pca_vectors_reduced)
    return cluster_labels


cluster_labels = kmeans_clustering(pca_vectors_reduced, optimal_k)

# Calculate the silhouette score
silhouette_avg = silhouette_score(pca_vectors_reduced, cluster_labels)
print(silhouette_avg)

# Assign the kmeans clusters to the original df and save df with clusters
df["kmean_clusters"] = cluster_labels

if not os.path.exists(os.path.join(BASE_PATH, 'clustered_data.csv')):
    df.to_csv(os.path.join(BASE_PATH, 'clustered_data.csv'), index=False)

# ----------------------------------------------------------------------------------------------------------------------
# GET THE TOP WORDS FOR EACH CLUSTER (USE TF-IDF VECTORS BECAUSE SHOWS UNIQUENESS OF WORDS IN DOCS)
# ----------------------------------------------------------------------------------------------------------------------


# get the top words for each cluster and save both in a separate table
def get_kmeans_top_words(cluster_num, df, tfidf_vectors, feature_names, n_top_words=20):
    cluster_indices = df[df["kmean_clusters"] == cluster_num].index
    cluster_scores = tfidf_vectors[cluster_indices].sum(axis=0).A1
    top_indices = cluster_scores.argsort()[::-1][:n_top_words]
    top_words = [feature_names[i] for i in top_indices]
    print(f"\n{cluster_num}\n{top_words}")
    return top_words


unique_clusters = df["kmean_clusters"].unique()
unique_clusters = sorted(unique_clusters, reverse=False)
cluster_top_words = []
for cluster_num in unique_clusters:
    top_words = get_kmeans_top_words(cluster_num, df, tfidf_vectors_final, tfidf_feature_names_final)
    cluster_top_words.append(top_words)

cluster_top_words_df = pd.DataFrame({"kmean_clusters": unique_clusters, "top_words": cluster_top_words})

if not os.path.exists(os.path.join(BASE_PATH, 'cluster_top_words.csv')):
    cluster_top_words_df.to_csv(os.path.join(BASE_PATH, 'cluster_top_words.csv'), index=False)
