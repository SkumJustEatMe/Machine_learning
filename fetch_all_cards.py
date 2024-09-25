import requests
import time
import pandas as pd

def fetch_all_sets():
    all_sets = []
    url = "https://api.scryfall.com/sets"
    
    while url:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            all_sets.extend([set_info['code'] for set_info in data['data']])
            
            # Check if there is another page
            url = data.get('next_page')
        else:
            print(f"Error: {response.status_code}")
            break
    
    return all_sets

def fetch_all_cards_in_set(url):
    all_cards = []
    
    while url:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            all_cards.extend(data['data'])  # Add the current page of cards to the list
            
            # Check if there is another page
            url = data.get('next_page')
        else:
            print(f"Error: {response.status_code}")
            break
    
    return all_cards

def save_cards_to_file(cards, filename):
    df = pd.DataFrame(cards)
    df.to_csv(filename, index=False)

# Main script
def main():
    full_list = []
    
    sets = fetch_all_sets()
    print(f"Fetched {len(sets)} sets.")
    
    for set_code in sets:
        cards = fetch_all_cards_in_set(f"https://api.scryfall.com/cards/search?order=set&q=e%3A{set_code}&unique=prints")
        full_list.extend(cards)
        print(f"Total cards fetched: {len(full_list)}")
        time.sleep(0.5)  # Respect rate limits
    
    # Save the collected cards to a CSV file
    save_cards_to_file(full_list, 'all_cards.csv')
    print("All cards saved to 'all_cards.csv'.")

# Run the script
if __name__ == "__main__":
    main()
