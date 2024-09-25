import pandas as pd
import ast
import warnings
import requests
import time
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

df = pd.read_csv("all_cards.csv")
pd.set_option('display.max_columns', None)

def get_data_set_info():
    print(f"Total number of cards: {len(df)}")
    print(df.columns)
    print(f"Id type: {type(df["id"])}\n\n {df.iloc[0]}")
    unique_langs = df['lang'].unique()
    print("Unique languages in the 'lang' column:")
    print(unique_langs)

    lang_counts = df['lang'].value_counts()
    print("\nCounts of each language in the 'lang' column:")
    print(lang_counts)

def remove_non_en_cards(dataframe):
    english_cards = dataframe[dataframe['lang'] == 'en']
    print(f"Removed Non english cards: {len(dataframe)-len(english_cards)}")
    print(f"EN cards: {len(english_cards)}")

    return english_cards

def remove_cards_without_price(dataframe):
    cards_with_price = []
    cards_without_price = []

    dataframe['prices'] = dataframe['prices'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    cards_with_price = dataframe[dataframe['prices'].apply(lambda x: pd.notna(x) and 'eur' in x and pd.notna(x['eur']))]
    cards_without_price = dataframe[~dataframe['prices'].apply(lambda x: pd.notna(x) and 'eur' in x and pd.notna(x['eur']))]

    print(f"cards with price: {len(cards_with_price)}")
    print(f"Rmoved cards without price: {len(cards_without_price)}")

    return cards_with_price

def save_cards_to_file(cards, filename):
    df = pd.DataFrame(cards)
    df.to_csv(filename, index=False)

def make_list_of_attribute(filename, attribute):
    df = pd.read_csv(filename)
    attribute_list = df[attribute].tolist()

    print(f"Csv file length: {len(df)}")
    print(f"Attribute List lenght: {len(attribute_list)}")
    return attribute_list

def fetch_card_prices_batch():
    url = "https://api.scryfall.com/cards/collection"
    card_prices = []
    card_ids = make_list_of_attribute("Golden_list.csv", "id")
    
    # Scryfall's /cards/collection allows up to 75 cards per request
    batch_size = 75
    total_fetched = 0
    
    for i in range(0, len(card_ids), batch_size):
        batch = card_ids[i:i + batch_size]
        request_payload = {
            "identifiers": [{"id": card_id} for card_id in batch]
        }
        
        response = requests.post(url, json=request_payload)
        
        if response.status_code == 200:
            cards_data = response.json().get('data', [])
            for card_data in cards_data:
                eur_price = card_data.get('prices', {}).get('eur')
                card_prices.append({
                    "id": card_data['id'],
                    "eur_price": eur_price
                })
                total_fetched += 1
        else:
            print(f"Failed to fetch batch starting at index: {i}")
        
        print(f"Number of fetched cards so far: {total_fetched}")
        time.sleep(0.1)  # Small delay to avoid hitting rate limits
    
    return card_prices

def compare_card_prices(old_file, new_file, output_file):
    # Load the old and new price files into DataFrames
    old_df = pd.read_csv(old_file)
    new_df = pd.read_csv(new_file)
    
    # Extract the 'eur' price from the 'prices' object in the old DataFrame
    old_df['eur_old'] = old_df['prices'].apply(lambda x: eval(x).get('eur') if pd.notnull(x) else None)
    
    # Merge the DataFrames on card ID with suffixes for the 'eur' columns
    merged_df = pd.merge(old_df[['id', 'eur_old']], new_df[['id', 'eur_price']], on='id', suffixes=('_old', '_new'))
    
    # Print the columns to verify their names
    print("Columns in merged DataFrame:", merged_df.columns)
    
    # Create a DataFrame to store results
    results = []
    
    # Compare prices and find changes
    for _, row in merged_df.iterrows():
        old_price = row.get('eur_old')
        new_price = row.get('eur_price')
        
        if old_price != new_price:
            results.append({
                'id': row['id'],
                'old_price': old_price,
                'new_price': new_price
            })
    
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results to a new CSV file
    results_df.to_csv(output_file, index=False)
    
    print(f"Comparison complete. Results saved to {output_file}")

def analyze_price_changes(file):
    df = pd.read_csv(file)
    
    if df.empty:
        print("The file is empty.")
        return
    
    changed_count = 0
    not_changed_count = 0
    
    for _, row in df.iterrows():
        old_price = row['old_price']
        new_price = row['new_price']
        
        if old_price != new_price:
            changed_count += 1
        else:
            not_changed_count += 1

    print(f"Number of objects where the price changed: {changed_count}")
    print(f"Number of objects where the price did not change: {not_changed_count}")

    

# df = remove_non_en_cards(df)
# df = remove_cards_without_price(df)

# save_cards_to_file(df, "Golden_list.csv")
# save_cards_to_file(fetch_card_prices_batch(), "Golden_list1.csv")
#compare_card_prices('Golden_list.csv', 'Golden_list1.csv', 'price_changes.csv')
analyze_price_changes('price_changes.csv')