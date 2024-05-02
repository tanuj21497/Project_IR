import os
import re
import pandas as pd

# Define the paths to the first and second folders
first_folder_path = 'first'
second_folder_path = 'second'
output_folder = 'output_folder'

# Function to extract numeric part from file name
def extract_numeric_part(file):
    match = re.search(r'\d+', file)
    if match:
        return int(match.group())
    return 0

# Get the list of CSV files in the first and second folders
first_files = sorted(os.listdir(first_folder_path), key=extract_numeric_part)
second_files = sorted(os.listdir(second_folder_path), key=extract_numeric_part)

# Iterate through the files in both folders
for i, (first_file, second_file) in enumerate(zip(first_files, second_files), start=1):
    # Load the CSV files from the first and second folders
    df1 = pd.read_csv(os.path.join(first_folder_path, first_file))
    df2 = pd.read_csv(os.path.join(second_folder_path, second_file))

    # Extract date from "BROADCAST DATE/TIME" column of first file
    df1['Date'] = pd.to_datetime(df1['BROADCAST DATE/TIME'], format='%d-%b-%Y %H:%M:%S').dt.strftime('%d-%m-%Y')

    # Extract date from "Date" column of second file
    df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%d-%m-%Y')

    # Function to find the next date at least 5 days later
    def find_next_date(date, df):
        current_index = df[df['Date'] == date].index[0]
        next_date_index = current_index + 5
        if next_date_index < len(df):
            return df.loc[next_date_index, 'Date']
        else:
            return None

    # Merge dataframes based on date
    merged_df = pd.merge(df1, df2, on='Date', how='inner')

    # Filter only the rows where the date in df2 is at least 5 days later
    merged_df['Next_Date'] = merged_df.apply(lambda row: find_next_date(row['Date'], df2), axis=1)

    # Calculate sentiment based on price change
    def calculate_sentiment(row):
        initial_price = row['Close']
        final_price = df2[df2['Date'] == row['Next_Date']]['Close'].iloc[0]  # Use the appropriate column name
        if final_price > initial_price * 1.05:
            return 1
        elif final_price < initial_price * 0.95:
            return -1
        else:
            return 0

    # Apply sentiment calculation
    merged_df['Sentiment'] = merged_df.apply(calculate_sentiment, axis=1)

    # Save the result to a new CSV file
    result_file_path = os.path.join(output_folder, f'result{i}.csv')
    merged_df.to_csv(result_file_path, index=False)

    print(f'Result saved to {result_file_path}')
