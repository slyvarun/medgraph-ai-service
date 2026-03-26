The error you're seeing happens because Mermaid is a bit "picky" about how it handles special characters (like colons, brackets, or spaces) inside node labels. In the previous version, the node for FastAPI or the Final Response likely triggered a syntax break because they weren't wrapped in quotes.I have fixed the Mermaid syntax below by using Node IDs (short tags like A, B, C) and wrapping the Labels in double quotes ["Text"]. This is the "bulletproof" way to write Mermaid for GitHub.Copy and Paste the section below into your README.md:Markdown# 🧬 MedGraph Nexus
### **Next-Gen GraphRAG for Pharmaceutical Intelligence**

[![Live Demo](https://img.shields.io/badge/Demo-Live_on_Render-00E676?style=for-the-badge&logo=render&logoColor=white)](https://medgraph-ai-service.onrender.com/)
[![Neo4j](https://img.shields.io/badge/Database-Neo4j_Graph-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Gemini 2.0](https://img.shields.io/badge/AI_Engine-Gemini_2.0-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

**MedGraph Nexus** is a specialized medicine-focused RAG (Retrieval-Augmented Generation) system. Unlike standard "Vector-only" models, MedGraph Nexus utilizes a **Knowledge Graph (Neo4j)** to provide 100% accurate, relationship-aware medical insights grounded in structured data.

---

## 🧠 Working Principle: The GraphRAG Advantage

MedGraph Nexus operates on a **Structured Retrieval Lifecycle**. This process ensures that every answer is grounded in factual graph nodes rather than "best-guess" text chunks.

### 1. The Intelligence Loop
* **Entity Extraction:** The query agent (Gemini) identifies the `Medicine Name`, `Manufacturer`, or `Category` from the user's natural language input.
* **Cypher Traversal:** Instead of a simple similarity search, the backend executes a **Graph Query** (Cypher) to pull the exact `:Medicine` node and its specific properties.
* **Context Synthesis:** The structured data (JSON) is passed to the LLM. Because the data is already labeled (e.g., `Manufacturer: Pfizer`), the AI provides grounded facts without "guessing" relationships.

### 2. The "Nexus" Fallback Hierarchy
To ensure high availability, the system follows a three-tier logic:
1.  **Tier 1: Local Graph Lookup** (Neo4j) - Instant, authoritative local data.
2.  **Tier 2: Global Registry Fallback** (**openFDA API**) - Fetches real-time drug labels if local data is missing.
3.  **Tier 3: Deterministic UI** - If the AI model is rate-limited, the system bypasses the LLM and serves a raw data table directly to the user.

---

## ⚔️ GraphRAG vs. Traditional Vector RAG

MedGraph Nexus solves the "Hallucination" problem common in standard AI models by moving from **Mathematical Similarity** to **Structural Logic**.

| Feature | Standard "Vector" RAG | **MedGraph Nexus (GraphRAG)** |
| :--- | :--- | :--- |
| **Logic** | Mathematical Similarity (Vector distance) | **Relational Logic** (Graph Traversal) |
| **Data Integrity** | High risk of mixing up drug facts | **Zero Cross-Contamination** of data |
| **Queries** | Struggles with "How many?" or "Compare X" | **Native Support** for property filtering |
| **Reliability** | Fails if the LLM is offline | **Deterministic Mode** works without AI |

---

## 🏗️ Technical Architecture

```mermaid
flowchart TD
    User(["User Query"]) --> API["FastAPI /ask"]
    API --> Agent["Query Agent"]

    subgraph Knowledge_Graph_Layer
    Agent --> Neo4j[("Neo4j Graph")]
    Neo4j -- No Match --> FDA["openFDA API"]
    end

    subgraph Intelligence_Layer
    Neo4j --> Gemini{"Gemini 2.0"}
    FDA --> Gemini
    Gemini -- Fail --> Det["Deterministic Formatter"]
    Gemini -- Success --> Resp["Markdown Response"]
    end

    Det --> Final["Final Response"]
    Resp --> Final
📊 Data Model (Neo4j Schema)The core entity is the :Medicine node, designed for high-speed indexing and property-based retrieval.Node PropertyTypeUsagenameStringPrimary Key (Indexed)categoryStringTherapeutic Class FilteringstrengthStringPrecise Dosage RetrievalmanufacturerStringBrand-specific QueriesindicationTextGrounding for AI Generation🚀 Deployment & Local Setup1. Environment Configuration (.env)Code snippetNEO4J_URI=neo4j+s://<your-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-password>
GEMINI_API_KEY=<your-key>
OPENFDA_API_KEY=<optional>
2. Local InstallationBash# Clone and Install
git clone [https://github.com/yourusername/medgraph-nexus.git](https://github.com/yourusername/medgraph-nexus.git)
pip install -r requirements.txt

# Ingest Medicine Data
python production_ingest.py --file medicine_dataset.csv --clear

# Launch API
uvicorn ai_service:app --host 0.0.0.0 --port 8000
🛡️ Security & ReliabilityHallucination Prevention: Context is passed as structured JSON, forcing the LLM to stick to the graph data.Health Monitoring: /health endpoint tracks Neo4j and API connectivity.Scalability: Optimized for Render with dynamic PORT binding and Gunicorn/Uvicorn support.👨‍💻 Developed BySai Varun Degala | Google Gemini Campus Ambassador LinkedIn | Project Link
