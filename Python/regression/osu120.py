import csv

def read_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        # Read and print each row of the CSV file
        for row in csv_reader:
            print(row)

# Usage example
read_csv('example.csv')