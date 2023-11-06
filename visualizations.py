"""
This is the third and final step of the project.
The following code contains visualizations of the results from clustering and and keyword extraction.
"""

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# Standard Library Imports
import os

# Third-Party Library Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


# ----------------------------------------------------------------------------------------------------------------------
# LOADING THE DATA
# ----------------------------------------------------------------------------------------------------------------------

# Load dataset with clusters and top words by cluster
BASE_PATH = '/YourFilePathHere/'
FILE_NAME_1 = 'cluster_top_words.csv'
FILE_PATH_1 = os.path.join(BASE_PATH, FILE_NAME_1)
FILE_NAME_2 = 'clustered_data.csv'
FILE_PATH_2 = os.path.join(BASE_PATH, FILE_NAME_2)

df_cluster_top_words = pd.read_csv(FILE_PATH_1, delimiter=',', encoding='utf-8')
df_clustered_data = pd.read_csv(FILE_PATH_2, delimiter=',', encoding='utf-8')


# ----------------------------------------------------------------------------------------------------------------------
# CREATE VISUALIZATIONS
# ----------------------------------------------------------------------------------------------------------------------

custom_colors = ['#0ABF4C', '#13427E', '#B3EFB2', '#1E81DF']

# ----------------------------------------------------------
# Word Clouds for each cluster's top words
# ----------------------------------------------------------

for index, row in df_cluster_top_words.iterrows():
    cluster = row['kmean_clusters']
    top_words_list = row['top_words']
    word_cloud = WordCloud(background_color='white', colormap="Paired").generate(top_words_list)
    plt.imshow(word_cloud)
    plt.title(f'Cluster {cluster} Word Cloud')
    plt.axis("off")
    plt.show()

# ----------------------------------------------------------
# Scatterplot showcasing the four clusters in relation to the pca vectors
# ----------------------------------------------------------
x_axis = df_clustered_data['pca_vector_1']
y_axis = df_clustered_data['pca_vector_2']
plt.figure(figsize=(8, 6))
plt.title('Scatterplot showcasing K-Means Clusters')
plt.xlabel('PCA Vector (1)')
plt.ylabel('PCA Vector (2)')
sns.scatterplot(x=x_axis, y=y_axis, hue=df_clustered_data.kmean_clusters, palette=custom_colors)
plt.show()

# ----------------------------------------------------------
# Bar chart with absolute distribution per cluster
# ----------------------------------------------------------
cluster_counts = df_clustered_data['kmean_clusters'].value_counts().sort_index()
plt.figure(figsize=(8, 4))
cluster_counts.plot(kind='bar', color='royalblue')
plt.title('Absolute Distribution of K-Means Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Occurrences')
plt.show()

# ----------------------------------------------------------
# Bar chart with absolute distribution per cluster grouped by created year
# ----------------------------------------------------------
# Count the occurrences of each cluster for each created_year
cluster_counts = df_clustered_data.groupby(['created_year', 'kmean_clusters']).size().unstack(fill_value=0)

# Create a bar chart for each created_year
plt.figure(figsize=(10, 6))
cluster_counts.plot(kind='bar', stacked=True, color=custom_colors)
plt.title('Absolute Distribution of K-Means Clusters Grouped by Created Year')
plt.xlabel('Created Year')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=0)
plt.legend(title='K-Means Cluster')
plt.show()

# ----------------------------------------------------------
# Line chart showing cluster growth/decline over the years
# ----------------------------------------------------------
# Create a line graph for each cluster
plt.figure(figsize=(10, 6))

for i, cluster in enumerate(cluster_counts.columns):
    plt.plot(cluster_counts.index, cluster_counts[cluster], label=f'Cluster {cluster}', color=custom_colors[i])

plt.title('Cluster Growth/Decline Over Years')
plt.xlabel('Created Year')
plt.ylabel('Number of Occurrences')
plt.legend(title='K-Means Cluster')
plt.show()