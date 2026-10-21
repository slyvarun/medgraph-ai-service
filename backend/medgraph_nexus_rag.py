"""
=============================================================================
MedGraph Nexus - Pharmaceutical Knowledge GraphRAG Pipeline
=============================================================================
Architecture:
  - Vector & Lexical Hybrid Search on Symptom nodes (via neo4j-graphrag)
  - Graph Traversal: (:Symptom)<-[:TREATS]-(:Ingredient)<-[:CONTAINS]-(:Brand)
  - LLM Synthesis: Google Cloud Vertex AI (Gemini 1.5 Pro)
  - Strictly relies on official `neo4j-graphrag` (No LangChain dependencies)
=============================================================================
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Official Neo4j Driver
from neo4j import GraphDatabase, Driver

# Official neo4j-graphrag components
from neo4j_graphrag.embeddings import VertexAIEmbeddings
from neo4j_graphrag.llm import VertexAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.generation import GraphRAG

# Load environment configuration (.env)
load_dotenv()


# ---------------------------------------------------------------------------
# 1. Neo4j Driver Initialization
# ---------------------------------------------------------------------------
def init_neo4j_driver() -> Driver:
    """
    Establish authenticated connection to Neo4j database using environment variables.
    """
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    if not neo4j_password:
        raise ValueError("CRITICAL: NEO4J_PASSWORD environment variable is not set.")

    # Initialize thread-safe Neo4j driver
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )
    return driver


# ---------------------------------------------------------------------------
# 2. Graph Schema & Retrieval Cypher Definition
# ---------------------------------------------------------------------------
# Schema Context:
# (:Brand)-[:CONTAINS]->(:Ingredient)-[:TREATS]->(:Symptom)
#
# HybridCypherRetriever matches the entrypoint Symptom node via hybrid search
# (Vector + Full-text) and exposes it as the bound variable `node`.
# We traverse backward: (node)<-[:TREATS]-(i:Ingredient)<-[:CONTAINS]-(b:Brand).
RETRIEVAL_QUERY = """
MATCH (node)<-[:TREATS]-(ingredient:Ingredient)
OPTIONAL MATCH (brand:Brand)-[:CONTAINS]->(ingredient)
WITH node.name AS Symptom,
     ingredient.name AS Composition,
     collect(DISTINCT brand.name) AS Alternative_Medicines,
     score
RETURN Symptom,
       Composition,
       Alternative_Medicines,
       score
ORDER BY score DESC
"""


# ---------------------------------------------------------------------------
# 3. GraphRAG Pipeline Factory
# ---------------------------------------------------------------------------
def build_medgraph_nexus_rag(driver: Driver) -> GraphRAG:
    """
    Constructs the end-to-end GraphRAG pipeline:
      1. VertexAIEmbeddings: Embeds user query with text-embedding-004.
      2. HybridCypherRetriever: Executes vector + full-text search on (:Symptom)
         and traverses backward to ingredients and brands.
      3. VertexAILLM: Grounded answer generation using Gemini 1.5 Pro.
      4. GraphRAG: Coordinates retrieval and generation steps.
    """
    # Step 3.1: Initialize Vertex AI Embedding Model
    # Dimensions: 768 for text-embedding-004
    embedder = VertexAIEmbeddings(
        model="text-embedding-004"
    )

    # Step 3.2: Initialize HybridCypherRetriever
    # Combines vector index (semantic similarity) and full-text index (exact/keyword matching)
    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name="symptom_vector_index",
        fulltext_index_name="symptom_text_index",
        embedder=embedder,
        retrieval_query=RETRIEVAL_QUERY
    )

    # Step 3.3: Initialize Vertex AI Generative LLM
    # Gemini 1.5 Pro handles grounded clinical reasoning and synthesis
    llm = VertexAILLM(
        model_name="gemini-1.5-pro"
    )

    # Step 3.4: Wrap Retriever and LLM inside GraphRAG Pipeline
    rag_pipeline = GraphRAG(
        retriever=retriever,
        llm=llm
    )

    return rag_pipeline


# ---------------------------------------------------------------------------
# 4. Test Execution Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("MedGraph Nexus: Hybrid GraphRAG Pipeline Initialization")
    print("=" * 80)

    # Acquire database driver
    driver = init_neo4j_driver()

    try:
        # Build the GraphRAG pipeline
        print("[1/3] Assembling Vertex AI Embedder, HybridCypherRetriever, & Gemini LLM...")
        rag = build_medgraph_nexus_rag(driver)
        print("[2/3] Pipeline successfully configured and ready.")

        # Test query representing a fuzzy clinical symptom description
        fuzzy_query = "my head is pounding"
        print(f"\n[3/3] Executing GraphRAG Search for Query: \"{fuzzy_query}\"")
        print("-" * 80)

        # Execute search across vector + fulltext indices, traverse graph, and synthesize
        response = rag.search(
            query_text=fuzzy_query,
            return_context=True  # Return contextual graph records alongside answer
        )

        print("\n=== MedGraph Nexus Clinical Response ===")
        print(response.answer)
        print("=" * 80)

        # Optionally print raw retrieved graph context for transparency
        if hasattr(response, "context") and response.context:
            print("\n[Retrieved Graph Knowledge Context]")
            print(response.context)

    except Exception as exc:
        print(f"\n[!] Pipeline Execution Error: {exc}", file=sys.stderr)
        raise exc

    finally:
        # Gracefully release database connections
        driver.close()
        print("\nNeo4j driver connection closed.")
