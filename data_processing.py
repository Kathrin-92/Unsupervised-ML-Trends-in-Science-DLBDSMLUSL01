"""
This code represents the initial step of the project.
Here, the data is imported and goes through a number of cleaning and pre-processing steps.
As necessary, some columns are added, changed in name, divided or deleted.
The result is a DataFrame that has  been cleaned and has all the necessary columns for further analysis.
"""

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# Standard Library Imports
import os
import json
import random
import re

# Third-Party Library Imports
import ast
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy

# Import the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Download NLTK data for tokenization
nltk.download('punkt')

# Download NLTK data for English stopwords
nltk.download('stopwords')

# Create a set of English stopwords using NLTK
english_stopwords = set(nltk.corpus.stopwords.words('english'))


# ----------------------------------------------------------------------------------------------------------------------
# LOADING THE DATA
# ----------------------------------------------------------------------------------------------------------------------

# Download original dataset here: https://www.kaggle.com/datasets/Cornell-University/arxiv/data
# Defining FILE_PATH variable
BASE_PATH = 'YourFilePathHere'
FILE_NAME = 'arxiv-metadata-oai-snapshot.json'
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)


# Utility function to yield data from the stored file
# Yield each line as a generator, allowing for efficient processing of large files
def extract_data(datapath):
    with open(datapath, 'r') as datafile:
        for line in datafile:
            yield line


# Utility function to yield N random records from the data generator and normalize them into a DataFrame
# Convert the data generator into a list of lines from the data file
# Calculate the number of records to load as a percentage of the total records
# Select a random subset of records based on the calculated count
# Parse and normalize the selected records as JSON and create a DataFrame
def fetch_random_normalized_records(data_gen, percentage_to_load):
    records = list(data_gen)
    num_records_to_load = int(len(records) * percentage_to_load)
    random_subset = random.sample(records, num_records_to_load)
    df = pd.json_normalize([json.loads(record) for record in random_subset])
    return df


# Extracting data from input file
data_gen = extract_data(FILE_PATH)
percentage_to_load = 0.10

# Define the name and path for the subset data file
data_file_name = 'data_subset.csv'
data_file_path = os.path.join(BASE_PATH, data_file_name)

# Check if the subset data file already exists
# If it doesn't exist, fetch a random subset of data, normalize it, and save it to the CSV file
if os.path.exists(data_file_path):
    df = pd.read_csv(data_file_path, delimiter=',', encoding='utf-8')
else:
    df = fetch_random_normalized_records(data_gen, percentage_to_load)
    df.to_csv(data_file_path, index=False)


# ----------------------------------------------------------------------------------------------------------------------
# FIRST PREPROCESSING AND CLEANING OF COLUMNS/ROWS
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------
# INSPECT THE DATA
# ----------------------------------------------------------
df.head()
df.info()
length_of_df = len(df)

# ----------------------------------------------------------
# HANDLE MISSING VALUES
# ----------------------------------------------------------

# drop entries with the same abstract and title
df.drop_duplicates(subset='title', keep='last', inplace=True)
df.drop_duplicates(subset='abstract', keep='last', inplace=True)

# check for missing data using isnull
df.isnull().sum()

# calculate the percentage share of NaN values in the total
df.isnull().sum()/length_of_df

# -->> Result:
# The columns 'report-no', 'license', 'journal-ref', 'doi' and 'comments' have a NaN share of 20-90%.
# This information is not decisive for the further processing of the question and is therefore dropped.

# drop columns with high percentage share of NaN values
df = df.drop(columns=['journal-ref', 'report-no', 'doi', 'license', 'comments'])

# ----------------------------------------------------------
# FORMAT DATA FURTHER & CREATE NEW AND/OR RENAME COLUMNS
# ----------------------------------------------------------

# ID:
# replace any character that is not a digit in ID so that it becomes a regular int value
df['id'] = df['id'].str.replace(r'[^0-9]','', regex=True).astype(int)

# VERSIONS
df['versions'] = df['versions'].apply(ast.literal_eval)
# Create new column to save the date where the first version was created as it is probably the publishing date
df['created_date_v1'] = df['versions'].apply(lambda x: [item['created'] for item in x
                                                        if item['version'] == 'v1'][0]
                                                        if any(item['version'] == 'v1' for item in x)
                                                        else None)
df['created_date_v1'] = pd.to_datetime(df['created_date_v1'], format='%a, %d %b %Y %H:%M:%S %Z')
df['created_date_v1'] = df['created_date_v1'].dt.tz_convert('UTC')
df = df.drop(['versions'], axis=1)

# create columns with month, year, and weekday
df['created_month'] = df['created_date_v1'].dt.month
df['created_year'] = df['created_date_v1'].dt.year
df['created_weekday'] = df['created_date_v1'].dt.day_name()

# consider the distribution of papers by year to define a possible time limit for the analysis
article_count_by_year = df.groupby('created_year')['id'].count().reset_index(name='count')

# bar chart of quantity distribution by year
plt.figure(figsize=(10, 4))
plt.bar(article_count_by_year['created_year'], article_count_by_year['count'], color='royalblue')
plt.title('Article Count by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# relatively few articles before 2010; because 2023 is the current year, it's also excluded from further consideration
# because still very large, we'll only consider the last five whole years, 2017-2022; length of df is now 96.378
df = df[(df['created_year'] > 2016) & (df['created_year'] != 2023)]

# CATEGORIES:
# split string to list and rename column
df['categories'] = df['categories'].str.split(' ')
df = df.rename(columns={"categories": "category_codes"})

# join with extracted df_arxiv_categories to get the names according to arxiv taxonomy
df_arxiv_categories = pd.read_csv(os.path.join(BASE_PATH, 'arxiv_taxonomy_metadata.csv'), delimiter=',',
                                  encoding='utf-8')
unique_category_groups = df_arxiv_categories['category_group'].unique()

for index, row in df.iterrows():
    category_names_list = []
    category_group_list = []
    for item in row['category_codes']:
        matching_rows = df_arxiv_categories[df_arxiv_categories['category_codes'] == item]
        if not matching_rows.empty:
            item_name = matching_rows['category_names'].iloc[0]
            group_name = matching_rows['category_group'].iloc[0]
            category_names_list.append(item_name)
            if group_name not in category_group_list:
                category_group_list.append(group_name)
    df.at[index, 'category_names'] = category_names_list
    df.at[index, 'category_group'] = category_group_list

# Flatten the list of category_group
categories = [category for sublist in df['category_group'] for category in sublist]

# Count the occurrences of each category
category_counts = pd.Series(categories).value_counts()
category_counts = category_counts[unique_category_groups].fillna(0)
category_counts_df = pd.DataFrame({'category_group': category_counts.index, 'count': category_counts.values})
category_counts_df = category_counts_df.sort_values(by='count', ascending=False)

# bar chart of quantity distribution by category_group based on raw data
plt.figure(figsize=(10, 10))
plt.bar(category_counts_df['category_group'], category_counts_df['count'], color='royalblue')
plt.title('Article Count by Category Group')
plt.xlabel('Category Group')
x_labels = [label.replace(' ', '\n') for label in category_counts_df['category_group']] # for line breaks
plt.xticks(range(len(x_labels)), x_labels, rotation=20)
plt.ylabel('Count')
plt.show()

# Count the number of records in more than one group
multiple_group_count = (df['category_group'].apply(lambda x: len(x)) > 1).sum()
# print("\nNumber of records in more than one group:", multiple_group_count) # >>> 19.487

# >>> Because of the quantity distribution in the groups and because the local memory does not have the required
# capacity to process the large amounts of data, only entries from the category
# "Quantitative Biology" are used in the following (1969 entries).
df = df[df['category_group'].apply(lambda x: 'Quantitative Biology' in x)]

# UPDATE DATE:
# convert to datetime format and save in separate columns; create colum with weekday
df['update_date'] = pd.to_datetime(df['update_date'])
df['update_month'] = df['update_date'].dt.month
df['update_year'] = df['update_date'].dt.year
df['update_weekday'] = df['update_date'].dt.day_name()

# TITLE:
# clean up: remove line breaks, convert to lowercase
df['title'] = df['title'].str.replace(r"\n", "")
df['title'] = df['title'].str.strip().astype(str)
df['title'] = df['title'].str.lower()

# AUTHORS
df['authors_parsed'] = df['authors_parsed'].apply(ast.literal_eval)


# Define a function to extract the last name of the first listed author
def extract_last_name(authors_list):
    if authors_list:
        return authors_list[0][0]  # Extract the first element's first element
    else:
        return None


# Apply the function to create the new column
df['author_last_name'] = df['authors_parsed'].apply(extract_last_name)
# get number of authors
df['number_of_authors'] = [len(i) for i in df['authors_parsed']]


# drop columns that are not needed for further analysis / project scope
df = df.drop(columns=['submitter', 'authors_parsed', 'authors', 'update_date', 'created_date_v1'])


# ----------------------------------------------------------------------------------------------------------------------
# CLEAN THE ABSTRACT COLUMN TO BE ABLE TO RUN A TF-IDF AND K-MEANS CLUSTERING
# ----------------------------------------------------------------------------------------------------------------------


# Define pre-cleaning function for abstract column
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters and punctuation
    text = re.sub(r"\s{2,}", " ", text) # Remove double white space
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text) # tokenize
    text = [token for token in text if token not in english_stopwords] # remove stopwords
    return text


# Apply the cleaning function to the 'abstract' column
df['abstract'] = df['abstract'].apply(lambda x: clean_text(x))
# Convert list of words to a single string
df['abstract'] = df['abstract'].apply(lambda x: ' '.join(x))


# Define the lemmatizing function
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(lemmatized_tokens)


# Apply the lemmatizing function to the pre-cleaned 'abstract' column
df['abstract_lemmatized'] = df['abstract'].apply(lambda x: lemmatize_text(x))

# For faster handling later on, the cleaned df is saved in a csv file
if not os.path.exists(os.path.join(BASE_PATH, 'data_subset_cleaned.csv')):
    df.to_csv(os.path.join(BASE_PATH, 'data_subset_cleaned.csv'), index=False)