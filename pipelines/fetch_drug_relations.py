"""
MedGraph Clinical Pipeline - openFDA & RxNorm Relationship Extraction
---------------------------------------------------------------------
Extracts brand names, active substances, indications, and RXCUI mappings from openFDA,
then resolves composition-level alternative brand names using NIH RxNorm allrelated.
"""

import os
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import quote

OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST/rxcui"
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")


def query_openfda_by_brand(brand_name: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Query openFDA /drug/label.json for a given brand name.
    Extracts brand_name, substance_name, raw indications_and_usage, and all product RXCUIs.
    """
    key = api_key or OPENFDA_API_KEY
    query_str = f'openfda.brand_name:"{brand_name}"'
    url = f"{OPENFDA_BASE_URL}?search={quote(query_str)}&limit=1"
    if key:
        url += f"&api_key={key}"

    headers = {"User-Agent": "MedGraph/1.0"}
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        return None

    data = response.json()
    results = data.get("results", [])
    if not results:
        return None

    item = results[0]
    openfda = item.get("openfda", {})

    # Extract required fields
    extracted_brand_names: List[str] = openfda.get("brand_name", [])
    extracted_substance_names: List[str] = openfda.get("substance_name", [])
    indications_raw_list: List[str] = item.get("indications_and_usage", [])
    raw_indications: str = "\n".join(indications_raw_list).strip()
    product_rxcuis: List[str] = openfda.get("rxcui", [])

    return {
        "brand_name": extracted_brand_names,
        "substance_name": extracted_substance_names,
        "indications_and_usage": raw_indications,
        "rxcui": product_rxcuis,
        "raw_openfda": openfda
    }


def get_rxnorm_all_related(rxcui: str) -> Optional[Dict[str, Any]]:
    """
    Query RxNorm /rxcui/{rxcui}/allrelated.json endpoint.
    """
    url = f"{RXNORM_BASE_URL}/{rxcui}/allrelated.json"
    headers = {"Accept": "application/json", "User-Agent": "MedGraph/1.0"}
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        return None
    return response.json()


def extract_concept_names_by_tty(rxnorm_data: Dict[str, Any], target_tty: str = "BN") -> List[str]:
    """
    Locates conceptGroup where tty matches target_tty and extracts the 'name' field for all concepts.
    """
    concept_groups = rxnorm_data.get("allRelatedGroup", {}).get("conceptGroup", [])
    names = []
    for group in concept_groups:
        if group.get("tty") == target_tty:
            for prop in group.get("conceptProperties", []):
                name = prop.get("name")
                if name and name not in names:
                    names.append(name)
    return names


def resolve_active_ingredient_rxcuis(rxnorm_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extracts active ingredient concept RXCUIs (tty: 'IN', 'MIN', 'PIN') from allrelated response.
    """
    concept_groups = rxnorm_data.get("allRelatedGroup", {}).get("conceptGroup", [])
    ingredients = []
    for group in concept_groups:
        if group.get("tty") in ("IN", "PIN", "MIN"):
            for prop in group.get("conceptProperties", []):
                ingredients.append({
                    "rxcui": prop.get("rxcui", ""),
                    "name": prop.get("name", ""),
                    "tty": group.get("tty")
                })
    return ingredients


def extract_alternative_brands_for_drug(brand_name: str) -> Dict[str, Any]:
    """
    End-to-end integration:
    1. Query openFDA /drug/label.json for brand_name.
    2. Extract brand_name, substance_name, raw indications_and_usage, and all product RXCUIs.
    3. Take active ingredient RXCUI provided by openFDA / RxNorm mapping.
    4. Query RxNorm /rxcui/{rxcui}/allrelated.json.
    5. Locate conceptGroup where tty is 'BN' and extract all name values.
    """
    fda_data = query_openfda_by_brand(brand_name)
    if not fda_data:
        return {"error": f"No label found for brand '{brand_name}' in openFDA"}

    product_rxcuis = fda_data["rxcui"]
    alternative_brands = []
    active_ingredients = []

    # Step 2: Use RXCUI mapping
    for cui in product_rxcuis[:3]:  # Check available product RXCUIs
        rx_related = get_rxnorm_all_related(cui)
        if not rx_related:
            continue

        # Look for active ingredient concepts (tty='IN')
        ingredients = resolve_active_ingredient_rxcuis(rx_related)
        for ing in ingredients:
            if ing not in active_ingredients:
                active_ingredients.append(ing)

        # Direct BN concepts from product CUI
        direct_bns = extract_concept_names_by_tty(rx_related, target_tty="BN")
        for b in direct_bns:
            if b not in alternative_brands:
                alternative_brands.append(b)

    # Step 3: Query using the active ingredient RXCUI to get ALL composition alternative brand names
    for ing in active_ingredients:
        ing_cui = ing["rxcui"]
        ing_related = get_rxnorm_all_related(ing_cui)
        if ing_related:
            ing_bns = extract_concept_names_by_tty(ing_related, target_tty="BN")
            for b in ing_bns:
                if b not in alternative_brands:
                    alternative_brands.append(b)

    return {
        "query_brand": brand_name,
        "openfda": {
            "brand_name": fda_data["brand_name"],
            "substance_name": fda_data["substance_name"],
            "indications_and_usage": fda_data["indications_and_usage"],
            "product_rxcuis": product_rxcuis
        },
        "active_ingredients": active_ingredients,
        "alternative_brand_names": alternative_brands
    }


if __name__ == "__main__":
    import json
    test_brand = "Lipitor"
    result = extract_alternative_brands_for_drug(test_brand)
    print("Extracted Brand Names:", result.get("openfda", {}).get("brand_name"))
    print("Extracted Substances:", result.get("openfda", {}).get("substance_name"))
    print("Product RXCUIs:", result.get("openfda", {}).get("product_rxcuis"))
    print("Active Ingredients:", result.get("active_ingredients"))
    print(f"Alternative Brand Names (tty='BN') ({len(result.get('alternative_brand_names', []))}):")
    print(result.get("alternative_brand_names", []))
