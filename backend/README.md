<div align="center">

  <img src="../frontend/logo.png" alt="MedGraph Nexus Logo" width="160" />

  # ⚕️ MedGraph Nexus Backend
  ### *Clinical Knowledge Graph & Multilingual GraphRAG Engine*

  [![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB_Cloud-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![GraphRAG](https://img.shields.io/badge/GraphRAG-Neo4j_%2B_LLM-FF6F00?style=for-the-badge&logo=ai&logoColor=white)](https://github.com/)

</div>

---

## 📁 Backend Directory Structure

```
backend/
├── .env                       # Environment variables & Neo4j credentials
├── .gitignore                 # Git ignore rules for virtual environments & secrets
├── ai_service.py              # FastAPI HTTP Server & Neo4j Keep-Alive engine
├── query_agent.py             # Multilingual GraphRAG reasoning agent (EN | TE | HI)
├── production_ingest.py       # Neo4j relational graph ingestion pipeline
├── medicine_dataset.csv       # 50,000 dataset records
├── medgraph_cache.json        # Embedded offline fallback graph cache
├── requirements.txt           # Python package dependencies
├── Procfile                   # Cloud deployment configuration
└── runtime.txt                # Python runtime specification
```

### Quick Execution:
```bash
python ai_service.py
```
Open **`http://localhost:8000`** in your browser.
