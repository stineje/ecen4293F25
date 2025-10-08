import csv

# Important Items
input_file_1 = 'Batting.csv'
input_file_2 = 'People.csv'
output_file = 'batting_avg_summary.csv'
ab_col = 'AB'
h_col = 'H'
year_col = 'yearID'
player_col = 'playerID'
name_key_col = 'playerID'
name_first_col = 'nameFirst'
name_last_col  = 'nameLast'
avg_col = 'average'
threshold_year = 1939
minimum_AB = 500
minimum_to_display = 25

try:
    # --- Load names ---
    names_dict = {}
    with open(input_file_2, mode='r', newline='', encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pid = row.get(name_key_col, '')
            if pid:
                names_dict[pid] = {
                    'first_name': row.get(name_first_col, '') or '',
                    'last_name':  row.get(name_last_col, '')  or ''
                }

    # --- Build per-season records (post-1939, AB >= 500) ---
    season_records = []  # (avg, playerID, first, last, year, AB, H)
    with open(input_file_1, mode='r', newline='', encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                year = int(row.get(year_col, '') or 0)
                ab   = float(row.get(ab_col, '') or 0.0)
                h    = float(row.get(h_col, '') or 0.0)
            except ValueError:
                continue  # skip bad rows

            if year >= threshold_year and ab >= minimum_AB:
                avg = h / ab if ab > 0 else 0.0
                pid = row.get(player_col, '')
                name = names_dict.get(pid, {'first_name': '', 'last_name': ''})
                season_records.append((avg, pid, name['first_name'], name['last_name'], year, int(ab), int(h)))

    # --- Sort and take Top N ---
    season_records.sort(key=lambda x: x[0], reverse=True)
    top_records = season_records[:minimum_to_display]

    if not top_records:
        print(f"No seasons found with AB >= {minimum_AB} where year >= {threshold_year}.")
    else:
        # --- Write CSV (now includes year) ---
        with open(output_file, mode='w', newline='', encoding='ISO-8859-1') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[player_col, name_first_col, name_last_col, year_col, 'AB', 'H', avg_col]
            )
            writer.writeheader()
            for avg, pid, first, last, year, ab, h in top_records:
                writer.writerow({
                    player_col: pid,
                    name_first_col: first,
                    name_last_col: last,
                    year_col: year,
                    'AB': ab,
                    'H': h,
                    avg_col: avg
                })

        # --- Print Top N with years ---
        print(f"Top {minimum_to_display} single-season batting averages (year >= {threshold_year}, AB ≥ {minimum_AB}):")
        for i, (avg, pid, first, last, year, ab, h) in enumerate(top_records, start=1):
            print(f"{i}: {avg:.8f} in {year} — {first} {last} (playerID: {pid}, {h}/{ab})")

except FileNotFoundError:
    print("Error: One of the files was not found.")
except KeyError as e:
    print(f"Error: The column '{e}' does not exist in one of the input files.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
