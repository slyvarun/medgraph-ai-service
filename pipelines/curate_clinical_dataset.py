import json
import re
import os

print("=" * 80)
print("MedGraph Curation Pipeline: Purging Non-Medicines & Consolidating Brand Repackagings")
print("=" * 80)

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

candidates = [
    os.path.join(parent_dir, "data", "medicine_dataset.json"),
    os.path.join(base_dir, "data", "medicine_dataset.json"),
    os.path.join(parent_dir, "medicine_dataset.json"),
    os.path.join(base_dir, "medicine_dataset.json"),
    "data/medicine_dataset.json",
    "medicine_dataset.json"
]

input_file = next((p for p in candidates if os.path.exists(p)), None)
if not input_file:
    print("Error: medicine_dataset.json not found in data/ or root.")
    exit(1)

output_file = input_file
print(f"Reading and writing dataset at: {input_file}")

with open(input_file, "r", encoding="utf-8") as f:
    meds = json.load(f)

total_raw = len(meds)
print(f"\n1. Loaded {total_raw:,} raw candidate records from {input_file}")

# 1. Non-medicine hygiene/cosmetic/sanitizer filter
EXCLUDED_KEYWORDS = [
    "hand sanitizer", "sanitizer", "mouthwash", "toothpaste", "dentifrice",
    "sunscreen", "body wash", "soap", "cleanser", "shampoo", "lip balm",
    "anti-aging", "wrinkle", "deodorant", "antiperspirant", "rubbing alcohol",
    "alcohol prep", "antiseptic wipe", "sterile water for irrigation",
    "saline nasal spray", "sterile saline flush", "skin cleanser"
]

def is_unnecessary(m):
    text = (m.get("brand_name", "") + " " + m.get("generic_name", "") + " " + m.get("indications", "")).lower()
    for kw in EXCLUDED_KEYWORDS:
        if kw in text:
            # Check if it's a legitimate prescription drug (e.g. prescription chlorhexidine)
            if m.get("product_type") == "HUMAN PRESCRIPTION DRUG":
                if any(x in text for x in ["hand sanitizer", "toothpaste", "sunscreen"]):
                    return True
                continue
            return True
    return False

# 2. Strict compound key for deduplication
def get_therapeutic_compound_key(m):
    subs = m.get("substances", [])
    if subs:
        norm_subs = sorted([re.sub(r'[^a-z0-9]', '', s.lower()) for s in subs if s.strip()])
        sub_key = "+".join(norm_subs)
    else:
        sub_key = ""
        
    gen = m.get("generic_name", "").lower()
    gen = re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ml|g|%|iu|meq)\b', '', gen)
    gen = re.sub(r'\b(tablet|tablets|capsule|capsules|injection|solution|oral|topical|cream|ointment|suspension|syrup|spray|patch|gel|powder|extended release|usp|hydrochloride|hcl|sodium|potassium)\b', '', gen)
    gen_key = re.sub(r'[^a-z0-9]', '', gen)
    
    route = m.get("route", "Oral").lower().strip()
    if any(r in route for r in ["oral", "mouth"]):
        route_norm = "oral"
    elif any(r in route for r in ["topical", "skin", "dermal"]):
        route_norm = "topical"
    elif any(r in route for r in ["intravenous", "intramuscular", "subcutaneous", "injection"]):
        route_norm = "injectable"
    elif any(r in route for r in ["inhalation", "respiratory", "nasal"]):
        route_norm = "inhalation"
    elif any(r in route for r in ["ophthalmic", "eye"]):
        route_norm = "ophthalmic"
    else:
        route_norm = "other"
        
    core = sub_key or gen_key
    if not core:
        brand = re.sub(r'[^a-z0-9]', '', m.get("brand_name", "").lower())
        return f"{brand}::{route_norm}"
        
    return f"{core}::{route_norm}"

clean_meds = []
removed_junk = []

for m in meds:
    if is_unnecessary(m):
        removed_junk.append(m)
    else:
        clean_meds.append(m)

print(f"2. Filtered out {len(removed_junk):,} non-medicine records (sanitizers, toothpastes, cosmetics, irrigation flushes).")
print(f"   Remaining legitimate clinical medicine records: {len(clean_meds):,}")

# 3. Consolidate duplicate brand repackagings
consolidated_medicines = {}

for m in clean_meds:
    ckey = get_therapeutic_compound_key(m)
    if ckey not in consolidated_medicines:
        master = dict(m)
        master["known_brands"] = [m["brand_name"]] if m["brand_name"] else []
        consolidated_medicines[ckey] = master
    else:
        master = consolidated_medicines[ckey]
        bname = m["brand_name"]
        if bname and bname not in master["known_brands"]:
            master["known_brands"].append(bname)
        # Update with longest and richest clinical narratives
        if len(m.get("indications", "")) > len(master.get("indications", "")):
            master["indications"] = m["indications"]
        if len(m.get("warnings", "")) > len(master.get("warnings", "")):
            master["warnings"] = m["warnings"]
        if len(m.get("adverse_reactions", "")) > len(master.get("adverse_reactions", "")):
            master["adverse_reactions"] = m["adverse_reactions"]
        if len(m.get("contraindications", "")) > len(master.get("contraindications", "")):
            master["contraindications"] = m["contraindications"]
        if len(m.get("drug_interactions", "")) > len(master.get("drug_interactions", "")):
            master["drug_interactions"] = m["drug_interactions"]
        if not master.get("rxcui") and m.get("rxcui"):
            master["rxcui"] = m["rxcui"]

final_list = list(consolidated_medicines.values())
duplicate_repackagings_merged = len(clean_meds) - len(final_list)

print(f"3. Consolidated {duplicate_repackagings_merged:,} duplicate brand repackagings.")
print(f"\n>>> FINAL VERIFIED PURE CLINICAL MEDICINES TO INJECT: {len(final_list):,}")

# Categorization and Product Type breakdown
pt_counts = {}
cat_counts = {}
for m in final_list:
    pt = m["product_type"]
    pt_counts[pt] = pt_counts.get(pt, 0) + 1
    cat = m["category"]
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

print("\n--- Final Breakdown by Product Type ---")
for pt, cnt in sorted(pt_counts.items(), key=lambda x: -x[1]):
    print(f"  • {pt}: {cnt:,} medicines")

print("\n--- Final Breakdown by Clinical Specialty ---")
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  • {cat}: {cnt:,} medicines")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_list, f, indent=2, ensure_ascii=False)

file_mb = round(os.path.getsize(output_file) / (1024 * 1024), 2)
print(f"\nSuccessfully wrote pristine master dataset to {output_file} ({file_mb} MB)")
print("=" * 80)
