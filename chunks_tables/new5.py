import tabula
import pandas as pd

# Replace 'your_file.pdf' with the path to your PDF file
file_path = '1.pdf'

# Use tabula to extract tables
tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)

# Save each table as a CSV file
for i, table in enumerate(tables):
    # Define the CSV file name
    csv_file_name = f"table_{i+1}.csv"
    
    # Save the table as a CSV file
    table.to_csv(csv_file_name, index=False)
    
    print(f"Table {i+1} saved as {csv_file_name}")
