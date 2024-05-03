import requests
from datetime import datetime

endpoint = "http://localhost:8000/patients/"

# Create a datetime object representing the birthday
birthday = datetime(year=2001, month=12, day=17)

# Format the birthday as a string in ISO 8601 format
birthday_iso = birthday.date().isoformat()

data = {"first_name": "Kareem", "last_name": "Hassan", "birthday": birthday_iso}

get_response = requests.post(endpoint, json=data)
print(get_response.text)
