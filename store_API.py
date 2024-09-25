import requests
import json
from datetime import datetime
import time

def fetch_all_cards_in_set(url):
    all_cards = []
    while url:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            all_cards.extend(data['data'])
            
            url = data.get('next_page')
            
            time.sleep(0.1)  # 100 milliseconds
        else:
            print(f"Error: {response.status_code}")
            break
    
    return all_cards

def load_existing_entries(file_path):
    existing_entries = set()
    
    try:
        with open(file_path, "r") as file:
            for line in file:
                entry = json.loads(line)
                existing_entries.add(entry['id'])  # Assuming 'id' is the unique identifier
    except FileNotFoundError:
        # If the file does not exist, return an empty set
        return existing_entries
    
    return existing_entries

def store_all(card_data, file_path):
    existing_entries = load_existing_entries(file_path)
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    with open(file_path, "a") as file:
        for entry in card_data:
            if entry['id'] not in existing_entries:
                # Add the date_str to the entry
                entry_with_date = entry.copy()
                entry_with_date['date'] = date_str
                
                json.dump(entry_with_date, file)
                file.write("\n")
                existing_entries.add(entry['id'])


# URL to fetch all cards in the DMU set
set_code = "dmu"
url = f"https://api.scryfall.com/cards/search?order=set&q=e%3A{set_code}&unique=prints"

cards = fetch_all_cards_in_set(url)
store_all(cards, "price_history.json")

for card in cards:
    print(f"{card['name']}")
