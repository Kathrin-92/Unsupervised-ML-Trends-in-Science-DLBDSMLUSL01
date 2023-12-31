Unsupervised Machine Learning 
# Transforming Text Data Into Insights: Analyzing Current Research Trends


## Table of Contents
1. [General Info](#General-Info)
2. [Installation](#Installation)


## General Info

**Project Overview** 

The field of scientific study is constantly changing, with new trends and topics of interest appearing on a regular basis. Identifying these trends is crucial for both scientific institutions and enterprises looking to navigate their strategic direction. This project aims to understand the current trends in science by analyzing scholarly publications using Natural Language Processing (NLP) techniques. The project is based on the arXiv database, which can be downloaded at https://www.kaggle.com/datasets/Cornell-University/arxiv/data. 


**Project Objective and Methods**

This project's primary goal is to use textual data analysis to extract insightful information and spot trends in scientific papers. The project uses NLP algorithms on the arXiv dataset in order to do this. The key steps are as follows::

1. Clustering Research Articles: Utilizing a k-Means clustering model, the project groups research articles with similar content. It employs Principal Component Analysis (PCA) for Dimensionality Reduction.
2. Topic Modeling Analysis: Term Frequency-Inverse Document Frequency (TF-IDF) is used to identify overarching subjects within the clusters.

The code for this project was developed as part of a university project for B.Sc. Data Science, with a focus on Unsupervised Learning and Feature Engineering.


**Results**

The project successfully identified four clusters of research articles and extracted the top 20 keywords for each cluster, providing insights into overarching themes and trends within the scientific publications.

![scatterplot-clusters](https://github.com/Kathrin-92/Unsupervised-ML-Trends-in-Science-DLBDSMLUSL01/assets/71875232/aa451a1a-46a0-44c1-a278-2757d1a2d61c)


**Key Skills Learned**

* Textual Data Analysis and NLP: The project involved analyzing and processing textual data from scientific publications, including techniques like TF-IDF and clustering.
* Machine Learning: Implementing unsupervised learning techniques such as k-Means clustering and PCA.
* Data Visualization: Creating visual representations of data using libraries like Matplotlib, Seaborn, and Yellowbrick.
* Data Cleaning and Preprocessing: Preparing and cleaning data for analysis, including lemmatization.


## Installation

**Requirements:** 

Make sure you have Python 3.7+ installed on your computer. You can download the latest version of Python [here](https://www.python.org/downloads/). 


**Req. Packages:**

matplotlib==3.6.2 <br>
pandas==1.5.2 <br>
requests==2.28.1 <br>
beautifulsoup4==4.11.1 <br>
nltk==3.7
numpy==1.23.5 <br>
spacy==3.5.3 <br>
scikit-learn==1.2.0 <br>
yellowbrick==1.5 <br>
seaborn==0.12.2 <br>
wordcloud==1.9.2 <br>
 
