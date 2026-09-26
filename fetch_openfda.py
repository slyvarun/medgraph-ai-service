import requests
import json
import os

# Get API key from environment or use directly for now
API_KEY = "8IV5urlxZUbtNRQs1lR8tLr1yBodJZOoogO7zphh"

def fetch_medicine_data(limit=100):
    print(f"Fetching {limit} records from OpenFDA...")
    url = f"https://api.fda.gov/drug/label.json?api_key={API_KEY}&search=_exists_:openfda.brand_name+AND+_exists_:indications_and_usage&limit={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        extracted_data = []
        for result in data.get('results', []):
            openfda = result.get('openfda', {})
            
            # Extract relevant fields
            brand_name = openfda.get('brand_name', [''])[0]
            generic_name = openfda.get('generic_name', [''])[0]
            manufacturer = openfda.get('manufacturer_name', [''])[0]
            route = openfda.get('route', [''])[0]
            product_type = openfda.get('product_type', [''])[0]
            
            # Get indications
            indications = result.get('indications_and_usage', [''])[0]
            
            if brand_name and indications:
                medicine = {
                    "brand_name": brand_name,
                    "generic_name": generic_name,
                    "manufacturer": manufacturer,
                    "route": route,
                    "product_type": product_type,
                    "indications": indications.strip()[:500] + "..." if len(indications) > 500 else indications.strip()
                }
                extracted_data.append(medicine)
        
        # Save to JSON
        with open('medicine_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4)
            
        print(f"Successfully saved {len(extracted_data)} records to medicine_dataset.json")
        return extracted_data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

if __name__ == "__main__":
    fetch_medicine_data(100)
