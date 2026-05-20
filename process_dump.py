import pandas as pd
import json
from tqdm import tqdm

def main():
    dump_filepath = '/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/ol_works_dump.txt' 
    raw_csv_path = '/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/data/books.csv'
    output_csv_path = '/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/data/enriched_books_from_dump.csv'

    print("1. Loading your local library...")
    df = pd.read_csv(raw_csv_path)
    
    # Ensure the column exists
    if 'api_description' not in df.columns:
        df['api_description'] = pd.NA

    # 2. CREATE THE HIGH-SPEED LOOKUP DICTIONARY
    # We convert titles to lowercase and strip whitespace so they match easily
    # Format: {"clean title": row_index}
    print("Building the memory map...")
    titles_to_find = {
        str(title).lower().strip(): idx 
        for idx, title in df['title'].items() 
        if pd.isna(df.at[idx, 'api_description'])
    }
    
    total_books = len(titles_to_find)
    matched_count = 0
    
    print(f"Hunting for {total_books} descriptions inside the 2.9GB dump...")

    # 3. STREAM THE DUMP (O(1) memory footprint)
    # Open Library format: type \t key \t revision \t last_modified \t JSON_DATA
    with open(dump_filepath, 'r', encoding='utf-8') as dump_file:
        for line in tqdm(dump_file, desc="Scanning 10GB Uncompressed Dump"):
            parts = line.split('\t')
            
            # Skip malformed lines
            if len(parts) < 5: 
                continue

            try:
                # The 5th column contains the actual JSON data
                json_string = parts[4]
                book_data = json.loads(json_string)
                
                # Clean the dump's title the exact same way
                dump_title = book_data.get('title', '').lower().strip()

                # 4. THE INSTANT MATCH
                if dump_title in titles_to_find:
                    desc = book_data.get('description', '')
                    
                    # Handle Open Library's messy JSON formats
                    if isinstance(desc, dict) and 'value' in desc:
                        desc = desc['value']
                        
                    if desc and isinstance(desc, str) and len(desc) > 20:
                        # Grab the index from our dictionary and update the dataframe
                        idx = titles_to_find[dump_title]
                        df.at[idx, 'api_description'] = desc
                        matched_count += 1
                        
                        # Remove it from our target list so we don't search for it anymore
                        del titles_to_find[dump_title]

                    # If we found all 6,800, stop reading the 2.9GB file!
                    if not titles_to_find:
                        print("\n🎉 Found every single book! Stopping the scan early.")
                        break
                        
            except json.JSONDecodeError:
                continue # Skip broken JSON lines
            except Exception as e:
                continue

    # 5. SAVE THE RESULTS
    print(f"\nScan complete. Successfully matched {matched_count} out of {total_books} books.")
    print("Saving to CSV...")
    df.to_csv(output_csv_path, index=False)
    print(f"Success! Your new dataset is waiting at {output_csv_path}")


if __name__ == "__main__":
    main()