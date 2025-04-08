import pandas as pd
import pickle


business = pd.read_json("data/yelp_academic_dataset_business.json", lines=True)

boba = business[business['categories'].str.contains('Bubble Tea', case=False, na=False)]

def total_hours_open(hours_dict):
    total = 0
    if not isinstance(hours_dict, dict):
        return 0  # skip if it's None or not a dict
    for time in hours_dict.values():
        try:
            open_time, close_time = time.split('-')
            open_hour, open_min = map(int, open_time.split(':'))
            close_hour, close_min = map(int, close_time.split(':'))

            open_minutes = open_hour * 60 + open_min
            close_minutes = close_hour * 60 + close_min

            # Handle overnight hours (e.g., 22:00-2:00)
            if close_minutes < open_minutes:
                close_minutes += 24 * 60

            total += (close_minutes - open_minutes) / 60
        except:
            continue
    return total

def latest_closing_time(hours_dict):
    latest = 0
    if not isinstance(hours_dict, dict):
        return None
    for time in hours_dict.values():
        try:
            close_hour, close_min = map(int, time.split('-')[1].split(':'))
            minutes = close_hour * 60 + close_min
            if minutes < 5 * 60:  # Treat closing times like 2:00am as next-day
                minutes += 24 * 60
            if minutes > latest:
                latest = minutes
        except:
            continue
    return latest / 60 if latest > 0 else None

def is_open_weekends(hours_dict):
    if not isinstance(hours_dict, dict):
        return 0
    return int('Saturday' in hours_dict and 'Sunday' in hours_dict)

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for day in days:
    col_name = f'open_{day.lower()}'
    boba[col_name] = boba['hours'].apply(lambda x: int(isinstance(x, dict) and day in x))


boba['total_hours_per_week'] = boba['hours'].apply(total_hours_open)
boba['latest_close_hour'] = boba['hours'].apply(latest_closing_time)
boba['open_7_days'] = boba['hours'].apply(lambda x: int(isinstance(x, dict) and len(x) == 7))
boba['open_weekends'] = boba['hours'].apply(is_open_weekends)

boba.to_csv('data/processed_boba_yelp_data.csv')