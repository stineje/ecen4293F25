import csv
from collections import defaultdict

# Important Items
input_file_1 = 'Batting.csv'
input_file_2 = 'People.csv'
output_file = 'batting_avg_summary.csv'
ab_col = 'AB'
h_col = 'H'
year_col = 'yearID'
player_col = 'playerID'
team_col = 'teamID'          # <-- optional: if present, we’ll list teams per season
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

    # --- Aggregate per (playerID, yearID) across all stints/teams ---
    # agg[(playerID, yearID)] = {'AB': int, 'H': int, 'teams': set()}
    agg = defaultdict(lambda: {'AB': 0, 'H': 0, 'teams': set()})
    with open(input_file_1, mode='r', newline='', encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                year = int(row.get(year_col, '') or 0)
                if year < threshold_year:
                    continue
                ab = int(float(row.get(ab_col, '') or 0.0))
                h  = int(float(row.get(h_col, '') or 0.0))
            except ValueError:
                continue  # skip bad rows

            pid = row.get(player_col, '')
            if not pid:
                continue

            key = (pid, year)
            agg[key]['AB'] += ab
            agg[key]['H']  += h

            # collect team IDs if present
            if team_col in row and row.get(team_col):
                agg[key]['teams'].add(row[team_col])

    # --- Build season records AFTER aggregation ---
    # (avg, playerID, first, last, year, AB, H, teams_str)
    season_records = []
    for (pid, year), stats in agg.items():
        ab_total = stats['AB']
        h_total  = stats['H']
        if ab_total >= minimum_AB:
            avg = h_total / ab_total if ab_total > 0 else 0.0
            name = names_dict.get(pid, {'first_name': '', 'last_name': ''})
            teams_str = ",".join(sorted(stats['teams'])) if stats['teams'] else ""
            season_records.append(
                (avg, pid, name['first_name'], name['last_name'], year, ab_total, h_total, teams_str)
            )

    # --- Sort and take Top N ---
    season_records.sort(key=lambda x: x[0], reverse=True)
    top_records = season_records[:minimum_to_display]

    if not top_records:
        print(f"No seasons found with AB >= {minimum_AB} where year >= {threshold_year}.")
    else:
        # --- Write CSV (includes year, AB/H totals, and teams if available) ---
        fieldnames = [player_col, name_first_col, name_last_col, year_col, 'AB', 'H', avg_col, 'teams']
        with open(output_file, mode='w', newline='', encoding='ISO-8859-1') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for avg, pid, first, last, year, ab, h, teams_str in top_records:
                writer.writerow({
                    player_col: pid,
                    name_first_col: first,
                    name_last_col: last,
                    year_col: year,
                    'AB': ab,
                    'H': h,
                    avg_col: avg,
                    'teams': teams_str
                })

        # --- Print Top N with years ---
        print(f"Top {minimum_to_display} single-season batting averages (year >= {threshold_year}, AB ≥ {minimum_AB}), aggregated across teams:")
        for i, (avg, pid, first, last, year, ab, h, teams_str) in enumerate(top_records, start=1):
            teams_note = f" | Teams: {teams_str}" if teams_str else ""
            print(f"{i}: {avg:.8f} in {year} — {first} {last} (playerID: {pid}, {h}/{ab}){teams_note}")

except FileNotFoundError:
    print("Error: One of the files was not found.")
except KeyError as e:
    print(f"Error: The column '{e}' does not exist in one of the input files.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    
