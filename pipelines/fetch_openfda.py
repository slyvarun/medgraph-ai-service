import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os
import re
import time

API_KEY = os.getenv("OPENFDA_API_KEY", "8IV5urlxZUbtNRQs1lR8tLr1yBodJZOoogO7zphh")

# Initialize resilient connection-pooled HTTP session
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=15, pool_maxsize=15))

# Clinical Specialty Keyword Mapper for Therapeutic Categorization
SPECIALTY_KEYWORDS = {
    "Cardiology & Vascular": ["hypertension", "cardiovascular", "arrhythmia", "angina", "heart", "statin", "beta blocker", "blood pressure", "vasodilator"],
    "Endocrinology & Diabetes": ["diabetes", "glycemic", "insulin", "metformin", "thyroid", "osteoporosis", "glucose", "hormone"],
    "Oncology & Hematology": ["cancer", "oncology", "neoplasm", "chemotherapy", "lymphoma", "leukemia", "melanoma", "tumor", "carcinoma"],
    "Antibiotics & Anti-Infectives": ["antibiotic", "antibacterial", "antiviral", "antifungal", "infection", "pneumonia", "bacterial", "microbial"],
    "Neurology & Pain": ["migraine", "epilepsy", "seizure", "parkinson", "neuropathy", "analgesic", "pain", "headache", "neurological"],
    "Psychiatry & Mental Health": ["antidepressant", "depression", "anxiety", "bipolar", "schizophrenia", "antipsychotic", "mood", "psychiatric"],
    "Pulmonology & Respiratory": ["asthma", "copd", "bronchodilator", "bronchospasm", "cough", "antihistamine", "respiratory", "inhalation"],
    "Gastroenterology & GI": ["acid reflux", "proton pump", "ulcer", "nausea", "vomiting", "diarrhea", "crohn", "gastrointestinal", "bowel", "antacid"],
    "Rheumatology & Autoimmune": ["arthritis", "anti-inflammatory", "nsaid", "immunosuppressive", "lupus", "gout", "joint"],
    "Dermatology & Allergy": ["dermatitis", "eczema", "psoriasis", "acne", "topical", "allergy", "allergic", "skin", "rash"],
    "Nephrology & Urology": ["diuretic", "kidney", "renal", "overactive bladder", "prostate", "urinary", "bladder"],
    "General & Emergency Medicine": ["anesthetic", "antidote", "emergency", "resuscitation", "fever", "fever reducer", "ophthalmic"]
}

def infer_therapeutic_category(indications: str, pharm_classes: list, default_type: str) -> str:
    """Infer clinical medical specialty based on indication narrative and drug classes."""
    text_to_check = (indications + " " + " ".join(pharm_classes)).lower()
    for specialty, kws in SPECIALTY_KEYWORDS.items():
        if any(kw in text_to_check for kw in kws):
            return specialty
    if default_type == "CELLULAR THERAPY":
        return "Cellular & Regenerative Medicine"
    return "General Medicine"

def clean_text(text: str, max_length: int = 400) -> str:
    """Normalize whitespace and truncate text safely."""
    if not text:
        return ""
    cleaned = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    if len(cleaned) > max_length:
        return cleaned[:max_length].rstrip() + "..."
    return cleaned

def get_canonical_key(brand_name: str, generic_name: str, substances: list) -> str:
    """
    Generate canonical key for compound-level deduplication.
    Strips dosage forms, strengths, numbers, and packaging affixes.
    """
    sub_key = substances[0].lower() if substances else ""
    gen_key = generic_name.lower() if generic_name and generic_name != "not specified" else ""
    brand_key = brand_name.lower()

    target = gen_key or sub_key or brand_key

    # Strip dosage forms, strengths, units
    cleaned = re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ml|g|%|iu|meq)\b', '', target)
    cleaned = re.sub(
        r'\b(tablet|tablets|capsule|capsules|injection|injections|solution|solutions|oral|topical|cream|ointment|suspension|syrup|spray|patch|gel|drop|drops|powder|extended release|delayed release|chewable|film coated|usp|hydrochloride|hcl|sodium|potassium)\b',
        '',
        cleaned
    )
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    cleaned = " ".join(cleaned.split())
    
    if len(cleaned) < 3:
        clean_brand = re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ml|g|%)\b', '', brand_key)
        clean_brand = " ".join(re.sub(r'[^a-z0-9\s]', ' ', clean_brand).split())
        return clean_brand or brand_key
        
    return cleaned

def fetch_openfda_labels(query: str, skip: int = 0, limit: int = 100):
    """Fetch paginated labels using pooled HTTPS session."""
    base_url = "https://api.fda.gov/drug/label.json"
    url = f"{base_url}?api_key={API_KEY}&search={requests.utils.quote(query)}&limit={limit}&skip={skip}"
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", [])
        return []
    except Exception as e:
        return []

def extract_record(item, default_product_type):
    """Extract clean medicine entity with rich clinical attributes."""
    openfda = item.get("openfda", {})
    brand_names = openfda.get("brand_name", [])
    generic_names = openfda.get("generic_name", [])
    manufacturers = openfda.get("manufacturer_name", [])
    routes = openfda.get("route", [])
    product_types = openfda.get("product_type", [])
    substances = openfda.get("substance_name", [])
    pharm_classes = openfda.get("pharm_class_epc", [])

    brand_name = brand_names[0].strip().title() if brand_names else ""
    if not brand_name:
        return None

    # Indications
    indications_raw = item.get("indications_and_usage", [""])[0]
    indications = clean_text(indications_raw, max_length=500)
    if not indications:
        indications = clean_text(item.get("purpose", [""])[0] or item.get("description", [""])[0], max_length=500)
    if not indications:
        return None

    clean_substances = [s.strip().title() for s in substances if s.strip()][:5]
    clean_classes = [c.replace("[EPC]", "").strip().title() for c in pharm_classes if c.strip()][:4]

    adverse = clean_text(item.get("adverse_reactions", [""])[0], max_length=350)
    warnings = clean_text(item.get("warnings", [""])[0] or item.get("boxed_warning", [""])[0], max_length=350)
    contra = clean_text(item.get("contraindications", [""])[0], max_length=350)
    interactions = clean_text(item.get("drug_interactions", [""])[0], max_length=350)

    generic_name = generic_names[0].strip().title() if generic_names else (clean_substances[0] if clean_substances else "Not Specified")
    product_type = product_types[0].strip() if product_types else default_product_type
    category = infer_therapeutic_category(indications, clean_classes, product_type)

    return {
        "brand_name": brand_name,
        "generic_name": generic_name,
        "product_type": product_type,
        "category": category,
        "manufacturer": manufacturers[0].strip() if manufacturers else "Unknown",
        "route": routes[0].strip() if routes else "Oral",
        "substances": clean_substances,
        "pharm_classes": clean_classes,
        "indications": indications,
        "warnings": warnings or "None specified in label summary",
        "adverse_reactions": adverse or "None specified in label summary",
        "contraindications": contra or "None specified in label summary",
        "drug_interactions": interactions or "None specified in label summary"
    }

def harvest_all_three_categories(output_file="medicine_dataset.json"):
    """
    Harvest medicines across all 3 official FDA categories:
    1. CELLULAR THERAPY (all 20 labels)
    2. HUMAN PRESCRIPTION DRUG (36,610 labels)
    3. HUMAN OTC DRUG (49,554 labels)
    """
    print("=" * 70, flush=True)
    print("MedGraph Harvester: Ingesting All 3 FDA Drug Categories", flush=True)
    print("  1. CELLULAR THERAPY        (20 labels)", flush=True)
    print("  2. HUMAN PRESCRIPTION DRUG (36,610 labels)", flush=True)
    print("  3. HUMAN OTC DRUG          (49,554 labels)", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    canonical_store = {}  # canon_key -> medicine_record

    # 1. Harvest CELLULAR THERAPY (all 20)
    print("\n--- [1/3] Harvesting CELLULAR THERAPY ---", flush=True)
    cell_items = fetch_openfda_labels('openfda.product_type.exact:"CELLULAR THERAPY"', skip=0, limit=100)
    for it in cell_items:
        rec = extract_record(it, "CELLULAR THERAPY")
        if rec:
            key = get_canonical_key(rec["brand_name"], rec["generic_name"], rec["substances"])
            canonical_store[key] = rec
    cell_count = len([r for r in canonical_store.values() if r['product_type'] == 'CELLULAR THERAPY'])
    print(f"Captured {cell_count} unique Cellular Therapies (Time: {round(time.time() - start_time, 1)}s)", flush=True)

    # 2. Priority Sweep: Top Essential First-Line Prescription & OTC Therapeutics
    print("\n--- Harvesting Essential First-Line Formularies (Metformin, Lisinopril, Statins, Antibiotics) ---", flush=True)
    essential_query = 'openfda.generic_name:("metformin" OR "lisinopril" OR "atorvastatin" OR "levothyroxine" OR "amlodipine" OR "omeprazole" OR "losartan" OR "albuterol" OR "gabapentin" OR "hydrochlorothiazide" OR "sertraline" OR "simvastatin" OR "montelukast" OR "escitalopram" OR "rosuvastatin" OR "bupropion" OR "furosemide" OR "pantoprazole" OR "duloxetine" OR "prednisone" OR "amoxicillin" OR "azithromycin" OR "doxycycline" OR "ciprofloxacin" OR "ibuprofen" OR "acetaminophen" OR "aspirin" OR "semaglutide" OR "empagliflozin" OR "dapagliflozin" OR "warfarin" OR "apixaban" OR "rivaroxaban")'
    essential_items = fetch_openfda_labels(f'_exists_:openfda.application_number AND ({essential_query})', skip=0, limit=100)
    ess_added = 0
    for it in essential_items:
        rec = extract_record(it, "HUMAN PRESCRIPTION DRUG")
        if rec:
            key = get_canonical_key(rec["brand_name"], rec["generic_name"], rec["substances"])
            if key not in canonical_store:
                canonical_store[key] = rec
                ess_added += 1
    print(f"Added {ess_added} core essential first-line medicines (Total unique: {len(canonical_store)})", flush=True)

    # 3. Harvest HUMAN PRESCRIPTION DRUG (A-Z partitions, 2 pages of 100 per letter = 200 labels/letter)
    print("\n--- [2/3] Harvesting HUMAN PRESCRIPTION DRUG (Rx Formularies A-Z) ---", flush=True)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for letter in alphabet:
        letter_start = time.time()
        new_in_letter = 0
        for page in range(2):
            skip = page * 100
            query = f'openfda.product_type.exact:"HUMAN PRESCRIPTION DRUG" AND openfda.brand_name:{letter}*'
            items = fetch_openfda_labels(query, skip=skip, limit=100)
            if not items:
                break
            for it in items:
                rec = extract_record(it, "HUMAN PRESCRIPTION DRUG")
                if not rec:
                    continue
                key = get_canonical_key(rec["brand_name"], rec["generic_name"], rec["substances"])
                if key not in canonical_store:
                    canonical_store[key] = rec
                    new_in_letter += 1
                else:
                    if len(rec["indications"]) > len(canonical_store[key]["indications"]):
                        canonical_store[key] = rec

        rx_so_far = len([r for r in canonical_store.values() if r['product_type'] == 'HUMAN PRESCRIPTION DRUG'])
        print(f"  • Rx Letter '{letter.upper()}': +{new_in_letter} unique compounds (Total Rx: {rx_so_far} | Total: {len(canonical_store)}) [{round(time.time() - letter_start, 1)}s]", flush=True)

    # 4. Harvest HUMAN OTC DRUG (A-Z partitions, 1 page of 100 per letter = 100 labels/letter)
    print("\n--- [3/3] Harvesting HUMAN OTC DRUG (Over-the-Counter Formularies A-Z) ---", flush=True)
    for letter in alphabet:
        letter_start = time.time()
        new_in_letter = 0
        query = f'openfda.product_type.exact:"HUMAN OTC DRUG" AND openfda.brand_name:{letter}*'
        items = fetch_openfda_labels(query, skip=0, limit=100)
        for it in items:
            rec = extract_record(it, "HUMAN OTC DRUG")
            if not rec:
                continue
            key = get_canonical_key(rec["brand_name"], rec["generic_name"], rec["substances"])
            if key not in canonical_store:
                canonical_store[key] = rec
                new_in_letter += 1
            else:
                if len(rec["indications"]) > len(canonical_store[key]["indications"]):
                    canonical_store[key] = rec

        otc_so_far = len([r for r in canonical_store.values() if r['product_type'] == 'HUMAN OTC DRUG'])
        print(f"  • OTC Letter '{letter.upper()}': +{new_in_letter} unique compounds (Total OTC: {otc_so_far} | Total: {len(canonical_store)}) [{round(time.time() - letter_start, 1)}s]", flush=True)

    dataset = list(canonical_store.values())
    elapsed = round(time.time() - start_time, 1)

    print("\n" + "=" * 70, flush=True)
    print(f"Extraction Completed in {elapsed} seconds!", flush=True)
    print(f"Total Unique Canonical Medicines: {len(dataset)}", flush=True)

    # Breakdown by Product Type
    pt_counts = {}
    spec_counts = {}
    for d in dataset:
        pt = d["product_type"]
        pt_counts[pt] = pt_counts.get(pt, 0) + 1
        spec = d["category"]
        spec_counts[spec] = spec_counts.get(spec, 0) + 1

    print("\nBreakdown by Product Type:")
    for pt, cnt in pt_counts.items():
        print(f"  • {pt}: {cnt} medicines")

    print("\nBreakdown by Therapeutic Specialty:")
    for spec, cnt in sorted(spec_counts.items(), key=lambda x: -x[1]):
        print(f"  • {spec}: {cnt} medicines")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved complete dataset to {output_file} ({round(os.path.getsize(output_file)/(1024*1024), 2)} MB)", flush=True)
    print("=" * 70, flush=True)
    return dataset

if __name__ == "__main__":
    harvest_all_three_categories()
