"""
This code block gets additional metadata from the arxiv website including the category groups and names.
"""

from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import os

# -----------------------------------------------------------------------------------------------------
# GET METADATA FROM ARXIV WEBSITE
# -----------------------------------------------------------------------------------------------------

# Make an HTTP GET request to the category taxonomy website
response = get('https://arxiv.org/category_taxonomy')
html_soup = BeautifulSoup(response.text, 'html.parser')

# Extract the category codes and names
category_codes = []
category_names = []
category_group = []
for h4 in html_soup.find_all('h4'):
    code, name = h4.text.strip().split(maxsplit=1)
    span = h4.find('span')
    name = span.text.strip('()') if span else ''
    category_codes.append(code)
    category_names.append(name)

for h2 in html_soup.find_all('h2', class_="accordion-head"):
    group_name = h2.text.strip()
    # print(group_name)
    category_group.append(group_name)

# Define the name and path for the metadata file
BASE_PATH = '/YourFilePathHere/'
data_file_name = 'arxiv_taxonomy_metadata.csv'
data_file_path = os.path.join(BASE_PATH, data_file_name)

# Check if the metadata file already exists
if not os.path.exists(data_file_path):
    # Save category codes and names in dataframe to merge with imported json data later on
    arxiv_categories = list(zip(category_codes, category_names))
    df_arxiv_categories = pd.DataFrame(arxiv_categories, columns=['category_codes', 'category_names'])
    df_arxiv_categories = df_arxiv_categories.iloc[1:]

    # function for assigning abbreviations to groups
    def map_category_group(category_code):
        if category_code.startswith('cs.'):
            return 'Computer Science'
        elif category_code.startswith('econ.'):
            return 'Economics'
        elif category_code.startswith('eess.'):
            return 'Electrical Engineering and Systems Science'
        elif category_code.startswith('math.'):
            return 'Mathematics'
        elif category_code.startswith('q-bio.'):
            return 'Quantitative Biology'
        elif category_code.startswith('q-fin.'):
            return 'Quantitative Finance'
        elif category_code.startswith('stat.'):
            return 'Statistics'
        else:
            return 'Physics'


    # apply the function to add the "category_group" column
    df_arxiv_categories['category_group'] = df_arxiv_categories['category_codes'].apply(map_category_group)
    df_arxiv_categories.to_csv(data_file_path, index=False)