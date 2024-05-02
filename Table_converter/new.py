import camelot
import pandas as pd

# Replace 'your_file.pdf' with the path to your PDF file
file_path = '1.pdf'

# Use Camelot to extract tables
tables = camelot.read_pdf(file_path, flavor='stream', pages='all')

# Save each table as a CSV file
for i, table in enumerate(tables):
    # Convert table data to DataFrame
    df = table.df
    
    # Save the DataFrame as CSV
    df.to_csv(f'table_{i+1}.csv', index=False)
    print(f"Table {i+1} saved as table_{i+1}.csv")
