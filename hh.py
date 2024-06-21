import pandas as pd
import calendar

# Define the start and end dates
start_date = '20230131'
end_date = '20231231'

# Function to adjust date to end of the month if not already end of the month
def adjust_to_end_of_month(date_str):
    date = pd.to_datetime(date_str, format='%Y%m%d')
    last_day_of_month = calendar.monthrange(date.year, date.month)[1]
    if date.day != last_day_of_month:
        date = date + pd.offsets.MonthEnd(0)
    return date

# Adjust the start date if necessary
start = adjust_to_end_of_month(start_date)
end = pd.to_datetime(end_date, format='%Y%m%d')

# Generate a date range with end of month frequency
date_range = pd.date_range(start=start, end=end, freq='M')

# Convert the date range to the desired format
end_of_month_dates = date_range.strftime('%Y%m%d').tolist()

# Print the result
for date in end_of_month_dates:
    print(date)
