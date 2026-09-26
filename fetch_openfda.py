import requests
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.getenv("OPENFDA_API_KEY", "8IV5urlxZUbtNRQs1lR8tLr1yBodJZOoogO7zphh")

def clean_text(text: str, max_length: int = 400) -> str:
    """Normalize whitespace and truncate narrative text."""
    if not text:
        return ""
    cleaned = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    if len(cleaned) > max_length:
        return cleaned[:max_length].rstrip() + "..."
    return cleaned

CATEGORIES = [
    {
        "name": "Cardiology",
        "query": 'indications_and_usage:("hypertension" OR "cardiovascular" OR "cardiac" OR "arrhythmia" OR "angina" OR "heart failure")'
    },
    {
        "name": "Diabetes",
        "query": 'indications_and_usage:("diabetes" OR "glycemic" OR "insulin" OR "metformin" OR "glucose" OR "hypoglycemia")'
    },
    {
        "name": "Oncology",
        "query": 'indications_and_usage:("cancer" OR "oncology" OR "neoplasm" OR "chemotherapy" OR "tumor" OR "leukemia" OR "lymphoma")'
    },
    {
        "name": "Antibiotics & Anti-Infectives",
        "query": 'indications_and_usage:("antibacterial" OR "antibiotic" OR "antiviral" OR "antifungal" OR "infection" OR "pneumonia")'
    },
    {
        "name": "General High-Demand Rx & OTC",
        "query": '_exists_:openfda.brand_name AND _exists_:indications_and_usage'
    }
]

def fetch_category_batch(category_name, search_query, skip, limit=100):
    """Fetch a single paginated batch for a given medical query."""
    base_url = "https://api.fda.gov/drug/label.json"
    url = f"{base_url}?api_key={API_KEY}&search=_exists_:openfda.brand_name+AND+({search_query})&limit={limit}&skip={skip}"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("results", [])
        return []
    except Exception as e:
        return []

def extract_medicine_record(item, default_category="General"):
    """Parse raw OpenFDA label into clean clinical medicine record."""
    openfda = item.get("openfda", {})
    brand_names = openfda.get("brand_name", [])
    generic_names = openfda.get("generic_name", [])
    manufacturers = openfda.get("manufacturer_name", [])
    routes = openfda.get("route", [])
    product_types = openfda.get("product_type", [])
    substances = openfda.get("substance_name", [])
    pharm_classes = openfda.get("pharm_class_epc", [])

    brand_name = brand_names[0].strip() if brand_names else ""
    if not brand_name:
        return None

    # Indications
    indications_raw = item.get("indications_and_usage", [""])[0]
    indications = clean_text(indications_raw, max_length=500)
    if not indications:
        return None

    # Clean active substances
    clean_substances = [s.strip().title() for s in substances if s.strip()][:5]
    clean_classes = [c.replace("[EPC]", "").strip().title() for c in pharm_classes if c.strip()][:4]

    # Clinical safety narratives
    adverse = clean_text(item.get("adverse_reactions", [""])[0], max_length=350)
    warnings = clean_text(item.get("warnings", [""])[0] or item.get("boxed_warning", [""])[0], max_length=350)
    contra = clean_text(item.get("contraindications", [""])[0], max_length=350)
    interactions = clean_text(item.get("drug_interactions", [""])[0], max_length=350)

    generic_name = generic_names[0].strip() if generic_names else (clean_substances[0] if clean_substances else "Not Specified")

    return {
        "brand_name": brand_name,
        "generic_name": generic_name,
        "category": default_category,
        "manufacturer": manufacturers[0].strip() if manufacturers else "Unknown",
        "route": routes[0].strip() if routes else "Oral",
        "product_type": product_types[0].strip() if product_types else "Prescription",
        "substances": clean_substances,
        "pharm_classes": clean_classes,
        "indications": indications,
        "warnings": warnings or "None specified in label summary",
        "adverse_reactions": adverse or "None specified in label summary",
        "contraindications": contra or "None specified in label summary",
        "drug_interactions": interactions or "None specified in label summary"
    }

def fetch_ultrascale_dataset(target_total=10000, output_file="medicine_dataset.json"):
    """
    Ultrascale data acquisition across Cardiology, Diabetes, Oncology,
    Antibiotics, and General medicines up to target_total records.
    """
    print(f"=== Starting Ultrascale Acquisition (Target: {target_total} medicines) ===")
    start_time = time.time()
    
    unique_medicines = {}
    
    # 1. Fetch targeted medical domains first to ensure top clinical representation
    for cat in CATEGORIES:
        if len(unique_medicines) >= target_total:
            break
            
        cat_name = cat["name"]
        cat_query = cat["query"]
        print(f"\n--- Harvesting Domain: {cat_name} ---")
        
        cat_quota = 2000 if cat_name != "General High-Demand Rx & OTC" else (target_total - len(unique_medicines))
        cat_harvested = 0
        skip = 0
        
        while cat_harvested < cat_quota and skip < 15000 and len(unique_medicines) < target_total:
            # Batch fetch with threads
            offsets = [skip + (i * 100) for i in range(5)]
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(fetch_category_batch, cat_name, cat_query, off): off for off in offsets}
                for f in as_completed(futures):
                    results = f.result()
                    for item in results:
                        rec = extract_medicine_record(item, default_category=cat_name)
                        if rec:
                            key = rec["brand_name"].lower()
                            if key not in unique_medicines:
                                unique_medicines[key] = rec
                                cat_harvested += 1
                                if len(unique_medicines) >= target_total:
                                    break
            skip += 500
            print(f"[{cat_name}] Collected {cat_harvested} medicines (Total unique so far: {len(unique_medicines)}/{target_total})")
            time.sleep(0.3)  # Respect rate limit

    dataset = list(unique_medicines.values())
    elapsed = round(time.time() - start_time, 1)
    print(f"\n=== Finished Ultrascale Extraction: {len(dataset)} unique medicines in {elapsed}s ===")
    
    # Breakdown by category
    breakdown = {}
    for d in dataset:
        c = d.get("category", "General")
        breakdown[c] = breakdown.get(c, 0) + 1
    for c, cnt in breakdown.items():
        print(f" - {c}: {cnt} medicines")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(dataset)} records to {output_file}")
    return dataset

if __name__ == "__main__":
    fetch_ultrascale_dataset(target_total=10000)
