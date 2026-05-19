import pandas as pd

def clean_and_prep_data(raw_data_path):
    # Load the raw data
    data = pd.read_csv(raw_data_path)

    # Select relevant columns
    df = data[['title', 'authors', 'average_rating', 'categories', 'published_year', 'ratings_count' ]]

    # Handle missing values
    df = df.dropna(subset=['title'])
    df['average_rating'] = df['average_rating'].fillna(0)
    df['ratings_count'] = df['ratings_count'].fillna(0)
    df['authors'] = df['authors'].fillna("Unknown Author")
    df['categories'] = df['categories'].fillna("Uncategorized")

    # Impute missing published years based on title
    missing_yrs = df[df['published_year'].isnull()]
    print(missing_yrs[['title' , 'authors']])
    
    df.loc[df['title'] == 'Book Club', 'published_year'] = 2012
    df.loc[df['title'] == 'The Civil War', 'published_year'] = 1958
    df.loc[df['title'] == 'The Fellowship of the Ring', 'published_year'] = 1954
    df.loc[df['title'] == 'Grade 4 Common Core Exemplar Collection', 'published_year'] = 2012
    df.loc[df['title'] == 'Frankenstein: City of Night: A Novel', 'published_year'] = 2005
    df.loc[df['title'] == '20000 LEAGUES UNDER THE SEA', 'published_year'] = 1870

    C = df['average_rating'].mean()
    m = df['ratings_count'].quantile(0.90)

    def calculate_weighted_rating(row, m, C):
        v = row['ratings_count']
        R = row['average_rating']
        if v == 0:
            return 0
        return (v / (v + m)) * R + (m / (v + m)) * C
    df['weighted_rating'] = df.apply(calculate_weighted_rating, axis=1, m=m, C=C)

    df['context_string'] = (
        "Title: " + df['title'] + ". " +
        "Author(s): " + df['authors'] + ". " +
        "Category: " + df['categories'] + ". " +
        "Published: " + df['published_year'].astype(int).astype(str) + ". " 
    )
    return df