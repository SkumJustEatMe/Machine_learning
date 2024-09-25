import requests

set_code = "dmu"
url = f"https://api.scryfall.com/cards/search?order=set&q=e%3A{set_code}&unique=prints"

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

# Fetch all cards in the DMU set
cards = fetch_all_cards_in_set(url)

# Print out the name and EUR prices of each card
for card in cards:
    if 'prices' in card:
        prices = card['prices']
        print(f"{card['name']}: EUR Price: {prices['eur']}")
        print(f"{card['name']}: EUR Foil Price: {prices['eur_foil']}")
    else:
        print(f"{card['name']}: No price data available.")















# # Check if the request was successful
# if response.status_code == 200:
#     # Parse the JSON response
#     card_data = response.json()

#     for item in card_data:
#         print(f"{item}: {card_data[item]}")
#     # # Print some details about the card
#     # print(f"Name: {card_data['name']}")
#     # print(f"Set: {card_data['set_name']}")
#     # print(f"Mana Cost: {card_data['mana_cost']}")
#     # print(f"Type Line: {card_data['type_line']}")
#     # print(f"Oracle Text: {card_data['oracle_text']}")
# else:
#     print(f"Failed to retrieve card data: {response.status_code}")

