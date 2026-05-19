from data_pipeline import clean_and_prep_data

cleaned_df = clean_and_prep_data('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/data/books_DL.csv')

cleaned_df.to_csv('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/data/books_DL.csv', index=False)
print("Data cleaning and preparation complete. Cleaned data saved to 'cleaned_books.csv'.")