import pandas as pd

def main():
    df = pd.read_csv('cleaned_books.csv')
    print (f'Loaded {len(df)} cleaned book records.')

if __name__ == "__main__":
    main()