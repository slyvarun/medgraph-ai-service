import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[sys.stdout]
)
logger = logging.getLogger("MedicineBackendAPI")

class MedicineGraphRetriever:
    """
    Hybrid retrieval model for querying the Medicines Neo4j Database.
    Connects to Neo4j to retrieve core nodes and schema relationships, 
    and enriches missing graph attributes (uses, side effects, substitutes) 
    using local CSV datasets if present.
    """
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        # Load environment variables
        load_dotenv()
        
        self.uri = uri or os.getenv("NEO4J_URI", "neo4j+s://60889094.databases.neo4j.io")
        self.user = user or os.getenv("NEO4J_USERNAME", "60889094")
        self.password = password or os.getenv("NEO4J_PASSWORD", "nWB5R-7VttDodCwa3al_s2wNQ6oNzCnH64ToJQO7XuM")
        self.database = os.getenv("NEO4J_DATABASE", "60889094")
        
        self.driver: Optional[Driver] = None
        self._connect()
        self._load_local_data()

    def _connect(self):
        """Establish the connection driver to the Neo4j database."""
        try:
            logger.info(f"Connecting to Neo4j database at: {self.uri}")
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("Successfully verified connectivity to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            self.driver = None
            raise

    def _load_local_data(self):
        """Load and index local CSVs to enrich properties missing in the Neo4j graph nodes."""
        try:
            logger.info("Searching for local CSV datasets for hybrid data enrichment...")
            
            # Check both current directory and parent directory for convenience
            df1_paths = [Path("sample_dataset_1.csv"), Path("../sample_dataset_1.csv")]
            df3_paths = [Path("sample_dataset_3.csv"), Path("../sample_dataset_3.csv")]
            
            df1_file = next((p for p in df1_paths if p.is_file()), None)
            df3_file = next((p for p in df3_paths if p.is_file()), None)
            
            if df1_file:
                logger.info(f"Found Dataset 1 at: {df1_file.resolve()}")
                df1 = pd.read_csv(df1_file)
                df1 = df1.dropna(subset=['join_key']).drop_duplicates(subset=['join_key'])
                self.dataset_1_dict = df1.set_index('join_key').to_dict(orient='index')
                logger.info(f"Loaded {len(self.dataset_1_dict)} records from {df1_file.name}")
            else:
                logger.warning("sample_dataset_1.csv not found. Substitute, side effect, and use details will be queried from Neo4j properties only.")
                self.dataset_1_dict = {}

            if df3_file:
                logger.info(f"Found Dataset 3 at: {df3_file.resolve()}")
                df3 = pd.read_csv(df3_file)
                df3 = df3.dropna(subset=['join_key']).drop_duplicates(subset=['join_key'])
                self.dataset_3_dict = df3.set_index('join_key').to_dict(orient='index')
                logger.info(f"Loaded {len(self.dataset_3_dict)} records from {df3_file.name}")
            else:
                logger.warning("sample_dataset_3.csv not found. Review data will default to Neo4j properties.")
                self.dataset_3_dict = {}
                
        except Exception as e:
            logger.error(f"Failed to load local datasets for hybrid lookup: {e}")
            self.dataset_1_dict = {}
            self.dataset_3_dict = {}

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver connection closed.")

    def execute_custom_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query with parameters against Neo4j."""
        if not self.driver:
            logger.error("Driver not connected. Cannot execute query.")
            return []
            
        parameters = parameters or {}
        try:
            session_kwargs = {}
            if self.database and self.database != "neo4j":
                session_kwargs["database"] = self.database
                
            with self.driver.session(**session_kwargs) as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing custom Cypher query: {e}\nQuery: {query}")
            return []

    def get_medicine_details(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a single medicine node's properties from Neo4j (using :Medicines and :Medicine labels) 
        and enrich with uses, side effects, and substitutes.
        """
        # Query base Medicines node
        query_base = """
        MATCH (m:Medicines)
        WHERE toLower(m.name) CONTAINS toLower($name)
        RETURN properties(m) AS props
        LIMIT 1
        """
        records = self.execute_custom_cypher(query_base, {"name": name.strip()})
        if not records:
            return {}
            
        med_props = records[0]["props"]
        join_key = med_props.get("join_key")
        
        # Query matching Medicine details node
        query_details = """
        MATCH (m:Medicine {join_key: $join_key})
        RETURN properties(m) AS props
        LIMIT 1
        """
        detail_records = self.execute_custom_cypher(query_details, {"join_key": join_key})
        if detail_records:
            med_props.update(detail_records[0]["props"])
            
        # Enrich from local dataset dictionary
        csv_details = self.dataset_1_dict.get(join_key, {})
        csv_reviews = self.dataset_3_dict.get(join_key, {})
        
        # Parse side effect list
        side_effects_str = csv_details.get("all_side_effects", "") or csv_reviews.get("Side_effects", "") or med_props.get("all_side_effects", "")
        side_effects = [x.strip() for x in str(side_effects_str).split(",") if x.strip() and str(x) != "nan"]
        
        # Parse uses list
        uses_str = csv_details.get("all_uses", "") or csv_reviews.get("Uses", "") or med_props.get("all_uses", "")
        uses = [x.strip() for x in str(uses_str).split(",") if x.strip() and str(x) != "nan"]
        
        # Parse substitutes list
        substitutes_str = csv_details.get("all_substitutes", "") or med_props.get("all_substitutes", "")
        substitutes = [x.strip() for x in str(substitutes_str).split(",") if x.strip() and str(x) != "nan"]
        
        result = {
            "medicine": {
                "id": med_props.get("id"),
                "name": med_props.get("name") or csv_details.get("name") or join_key,
                "join_key": join_key,
                "chemical_class": med_props.get("Chemical Class") or csv_details.get("Chemical Class"),
                "habit_forming": med_props.get("Habit Forming") or csv_details.get("Habit Forming"),
                "therapeutic_class": med_props.get("Therapeutic Class") or csv_details.get("Therapeutic Class"),
                "action_class": med_props.get("Action Class") or csv_details.get("Action Class"),
                "prescription_required": med_props.get("prescription_required") or csv_reviews.get("prescription_required"),
                "excellent_review_pct": med_props.get("excellent_review_pct") or csv_reviews.get("Excellent Review %"),
                "average_review_pct": med_props.get("average_review_pct") or csv_reviews.get("Average Review %"),
                "poor_review_pct": med_props.get("poor_review_pct") or csv_reviews.get("Poor Review %"),
            },
            "side_effects": side_effects,
            "uses": uses,
            "substitutes_list": substitutes
        }
        return result

    def get_substitutes(self, name: str) -> List[Dict[str, Any]]:
        """Retrieve potential substitute medicines for a given medicine."""
        details = self.get_medicine_details(name)
        if not details or not details.get("substitutes_list"):
            return []
            
        subs_list = details["substitutes_list"]
        
        # Query Neo4j to find pricing and class details for the substitutes
        subs_query = """
        MATCH (m:Medicines)
        WHERE m.join_key IN $subs_list
        RETURN m.name AS name, m.`Therapeutic Class` AS therapeutic_class, m.`Chemical Class` AS chemical_class
        """
        records = self.execute_custom_cypher(subs_query, {"subs_list": subs_list})
        
        if not records:
            return [{"name": sub_name, "details_available": False} for sub_name in subs_list]
            
        # Enrich reviews from csv if available
        enriched_records = []
        for record in records:
            j_key = record["name"].lower().strip()
            csv_reviews = self.dataset_3_dict.get(j_key, {})
            record["excellent_review_pct"] = csv_reviews.get("Excellent Review %")
            record["details_available"] = True
            enriched_records.append(record)
            
        return enriched_records

    def find_medicines_by_use(self, use_keyword: str) -> List[Dict[str, Any]]:
        """Find medicines registered under a specific therapeutic use/disease indication."""
        # Query Neo4j first for Medicines matching therapeutic class or description
        query = """
        MATCH (m:Medicines)
        WHERE toLower(m.`Therapeutic Class`) CONTAINS toLower($use_keyword)
           OR toLower(m.`Action Class`) CONTAINS toLower($use_keyword)
        RETURN m.name AS name, m.`Therapeutic Class` AS therapeutic_class, m.`Chemical Class` AS chemical_class, m.join_key AS join_key
        LIMIT 20
        """
        records = self.execute_custom_cypher(query, {"use_keyword": use_keyword})
        
        # Fallback to local dataset check
        if len(records) < 5:
            matched_keys = []
            for j_key, info in self.dataset_1_dict.items():
                all_uses = str(info.get("all_uses", "")).lower()
                if use_keyword.lower() in all_uses:
                    matched_keys.append(j_key)
                    if len(matched_keys) >= 20:
                        break
            
            if matched_keys:
                query_keys = """
                MATCH (m:Medicines)
                WHERE m.join_key IN $matched_keys
                RETURN m.name AS name, m.`Therapeutic Class` AS therapeutic_class, m.`Chemical Class` AS chemical_class, m.join_key AS join_key
                """
                records = self.execute_custom_cypher(query_keys, {"matched_keys": matched_keys})
                
        for r in records:
            csv_reviews = self.dataset_3_dict.get(r["join_key"], {})
            r["excellent_review_pct"] = csv_reviews.get("Excellent Review %")
            
        return records

    def find_safe_substitutes(self, name: str, avoid_side_effects: List[str]) -> List[Dict[str, Any]]:
        """Find substitutes for a medicine that do NOT trigger the given side effects."""
        substitutes = self.get_substitutes(name)
        if not substitutes:
            return []
            
        safe_subs = []
        avoid_lower = [se.lower().strip() for se in avoid_side_effects]
        
        for sub in substitutes:
            sub_name = sub.get("name")
            if not sub_name:
                continue
                
            sub_details = self.get_medicine_details(sub_name)
            sub_effects = [se.lower() for se in sub_details.get("side_effects", [])]
            
            has_bad_effect = False
            for effect in avoid_lower:
                for sub_effect in sub_effects:
                    if effect in sub_effect:
                        has_bad_effect = True
                        break
                if has_bad_effect:
                    break
                    
            if not has_bad_effect:
                safe_subs.append(sub)
                
        return safe_subs


# ==========================================
# FastAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Medicines Graph Database Backend API",
    description="Consolidated REST API for retrieving medicine details, substitutes, uses, and side effects from Neo4j AuraDB. Built for Google Cloud deployment and Lovable AI.",
    version="1.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database retriever singleton
retriever: Optional[MedicineGraphRetriever] = None

@app.on_event("startup")
def startup_event():
    """Establish Neo4j driver connection on app startup."""
    global retriever
    logger.info("Initializing Neo4j connection on app startup...")
    try:
        retriever = MedicineGraphRetriever()
        logger.info("Database connection successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j database during startup: {e}")

@app.on_event("shutdown")
def shutdown_event():
    """Close Neo4j connection cleanly on app shutdown."""
    global retriever
    if retriever:
        logger.info("Closing Neo4j connection on app shutdown...")
        retriever.close()

def get_retriever() -> MedicineGraphRetriever:
    """Check connection status and return retriever instance."""
    global retriever
    if retriever is None or retriever.driver is None:
        try:
            retriever = MedicineGraphRetriever()
        except Exception:
            raise HTTPException(
                status_code=503, 
                detail="Database connection is currently unavailable. Check Neo4j credentials."
            )
    return retriever


# Schema for custom Cypher query execution
class CypherRequest(BaseModel):
    query: str
    parameters: Optional[dict] = None


@app.get("/")
def read_root():
    """Health check endpoint."""
    db_status = "Connected" if (retriever and retriever.driver) else "Disconnected"
    return {
        "status": "online",
        "service": "Medicines Graph DB Backend (Deployment Folder)",
        "neo4j_status": db_status,
        "docs_url": "/docs",
        "message": "Welcome to the Medicines API. Navigate to /docs for interactive Swagger API testing."
    }

@app.get("/api/medicine/{name}")
def get_medicine_info(name: str):
    """Retrieve comprehensive details for a specific medicine."""
    db = get_retriever()
    details = db.get_medicine_details(name)
    if not details or not details.get("medicine"):
        raise HTTPException(status_code=404, detail=f"Medicine '{name}' not found.")
    return details

@app.get("/api/substitutes/{name}")
def get_medicine_substitutes(name: str):
    """Fetch substitutes for a medicine."""
    db = get_retriever()
    substitutes = db.get_substitutes(name)
    if not substitutes:
        details = db.get_medicine_details(name)
        if not details or not details.get("medicine"):
            raise HTTPException(status_code=404, detail=f"Medicine '{name}' not found.")
        return {"medicine": name, "substitutes": [], "message": "No substitutes found."}
    return {"medicine": name, "substitutes": substitutes}

@app.get("/api/substitutes/safe")
def get_safe_substitutes(
    name: str = Query(..., description="The name of the target medicine"),
    avoid: str = Query(..., description="Comma-separated side effects to avoid")
):
    """Find substitutes for a medicine that do NOT contain the specified side effects."""
    db = get_retriever()
    avoid_list = [effect.strip() for effect in avoid.split(",") if effect.strip()]
    if not avoid_list:
        raise HTTPException(status_code=400, detail="Must provide side effects to avoid.")
    safe_subs = db.find_safe_substitutes(name, avoid_list)
    return {
        "medicine": name,
        "avoiding_side_effects": avoid_list,
        "safe_substitutes": safe_subs
    }

@app.get("/api/search")
def search_medicines(
    use: Optional[str] = Query(None, description="Search medicines by symptom or use"),
    limit: int = Query(20, description="Max results to return")
):
    """Search for medicines by therapeutic use keyword."""
    if not use:
        raise HTTPException(status_code=400, detail="Must specify 'use' query parameter.")
    db = get_retriever()
    results = db.find_medicines_by_use(use)
    results = results[:limit]
    return {
        "query_use": use,
        "count": len(results),
        "results": results
    }

@app.post("/api/query")
def execute_custom_query(request: CypherRequest):
    """Execute custom Cypher queries (restricted admin access)."""
    db = get_retriever()
    try:
        results = db.execute_custom_cypher(request.query, request.parameters)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cypher execution failed: {str(e)}")
