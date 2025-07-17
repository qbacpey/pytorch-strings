import csv

# --- Configuration ---
CSV_FILE_1 = "0006_unsort.csv"
CSV_FILE_2 = "0004_unsorted_cuda_cpu.csv"
OUTPUT_CSV = "0007_all_cuda_cpu.csv"

# --- Merge Logic ---
with open(CSV_FILE_1, newline='', encoding='utf-8') as f1, \
     open(CSV_FILE_2, newline='', encoding='utf-8') as f2, \
     open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as fout:

    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    writer = csv.writer(fout)

    # Read and write the header from the first file
    header = next(reader1)
    writer.writerow(header)

    # Write the rows from the first file
    for row in reader1:
        writer.writerow(row)

    # Skip the header in the second file
    next(reader2)

    # Write the rows from the second file
    for row in reader2:
        writer.writerow(row)

print(f"Merged '{CSV_FILE_1}' and '{CSV_FILE_2}' into '{OUTPUT_CSV}'")