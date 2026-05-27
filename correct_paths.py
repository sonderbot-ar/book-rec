import os
from data.data_pipeline import clean_and_prep_data

# 1. Create a dynamic anchor to the folder this script is sitting in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build the paths dynamically
input_file = os.path.join(BASE_DIR, 'data', 'books.csv')
output_file = os.path.join(BASE_DIR, 'data', 'cleaned_books.csv')

# 3. Run the pipeline
cleaned_df = clean_and_prep_data(input_file)
cleaned_df.to_csv(output_file, index=False)