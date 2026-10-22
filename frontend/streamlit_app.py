"""
MedGraph Nexus - Streamlit Community Cloud Interface
-----------------------------------------------------
Interactive Clinical Graph Explorer and Symptom Search with Network Visualization.
Supports:
  - Natural language fuzzy symptom searching
  - Interactive knowledge graph rendering with color-coded nodes
  - Direct brand-to-ingredient alternative medicine lookup
  - Medical safety disclaimers & clinical guardrails
"""

import os
import requests
import streamlit as st

# Configure Streamlit page layout & styling
st.set_page_config(
    page_title="MedGraph Nexus — Clinical Graph Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint Configuration
API_BASE_URL = os.getenv("MEDGRAPH_API_URL", "http://localhost:8008")

# Custom CSS for modern clinical styling
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background: #ffffff; padding: 12px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .clinical-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        margin-bottom: 15px;
    }
    .badge-symptom { background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 6px; font-size: 0.85em; font-weight: 600; }
    .badge-ing { background-color: #dbeafe; color: #1e40af; padding: 4px 8px; border-radius: 6px; font-size: 0.85em; font-weight: 600; }
    .badge-brand { background-color: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 6px; font-size: 0.85em; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Controls & Clinical Metadata
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=64)
    st.title("MedGraph Nexus")
    st.caption("Pharmaceutical Knowledge Graph & Hybrid GraphRAG")
    st.divider()

    st.subheader("⚙️ Configuration")
    api_url = st.text_input("Backend API URL", value=API_BASE_URL)
    language = st.selectbox("Response Language", ["en (English)", "es (Spanish)", "fr (French)", "de (German)", "hi (Hindi)", "te (Telugu)"], index=0)
    lang_code = language.split()[0]

    st.divider()
    st.markdown("### 🏷️ Graph Legend")
    st.markdown("- 🔴 **Symptom Node**: Clinical indication")
    st.markdown("- 🔵 **Ingredient Node**: Active chemical composition")
    st.markdown("- 🟢 **Brand Node**: Marketed medicine formulation")

    st.divider()
    st.info("🛡️ **Medical Guardrail Active**: All generated responses strictly grounded in verified OpenFDA and RxNorm graphs.")

# ---------------------------------------------------------------------------
# Main App Header
# ---------------------------------------------------------------------------
st.title("🧬 MedGraph Nexus: Clinical Symptom & Drug Explorer")
st.markdown("Map colloquial symptom queries to pharmacological active ingredients and explore alternative brand formulations.")

tab1, tab2 = st.tabs(["🔍 Symptom Search (GraphRAG)", "💊 Direct Brand Alternatives Lookup"])

# ---------------------------------------------------------------------------
# Helper: Graph Visualizer
# ---------------------------------------------------------------------------
def render_network_graph(graph_data):
    """Render interactive network graph using streamlit-agraph or fallback HTML/Pyvis."""
    if not graph_data or not graph_data.get("nodes"):
        st.warning("No graph connections found for this inquiry.")
        return

    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        nodes = []
        node_colors = {
            "Symptom": "#ef4444",      # Coral Red
            "Ingredient": "#3b82f6",   # Sapphire Blue
            "Brand": "#10b981"         # Emerald Green
        }

        for n in graph_data["nodes"]:
            ntype = n.get("type", "Brand")
            color = node_colors.get(ntype, "#6b7280")
            nodes.append(Node(
                id=n["id"],
                label=n["label"],
                size=22 if ntype == "Symptom" else 18,
                color=color,
                font={"color": "#111827", "size": 13, "strokeWidth": 2, "strokeColor": "#ffffff"}
            ))

        edges = []
        for e in graph_data["edges"]:
            edges.append(Edge(
                source=e["source"],
                target=e["target"],
                label=e.get("relationship", ""),
                color="#94a3b8"
            ))

        config = Config(
            width=850,
            height=480,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#fbbf24"
        )
        agraph(nodes=nodes, edges=edges, config=config)

    except ImportError:
        # Fallback to Graphviz if streamlit-agraph is not yet installed
        import graphviz
        dot = graphviz.Digraph(comment="MedGraph Nexus")
        dot.attr(rankdir="LR", bgcolor="transparent")

        color_map = {"Symptom": "#ef4444", "Ingredient": "#3b82f6", "Brand": "#10b981"}
        for n in graph_data["nodes"]:
            c = color_map.get(n.get("type"), "#6b7280")
            dot.node(n["id"], label=f"{n['label']}\n({n.get('type')})", shape="box", style="filled,rounded", fillcolor=c, fontcolor="white")

        for e in graph_data["edges"]:
            dot.edge(e["source"], e["target"], label=e.get("relationship", ""))

        st.graphviz_chart(dot)


# ---------------------------------------------------------------------------
# TAB 1: Natural Language Symptom Search
# ---------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        query_input = st.text_input(
            "Enter symptom or clinical complaint:",
            value="my head is pounding and throbbing",
            placeholder="e.g. chronic cough, high blood pressure, pounding headache..."
        )
    with col2:
        st.write("")
        st.write("")
        search_clicked = st.button("🔎 Search Graph", use_container_width=True, type="primary")

    if search_clicked or query_input:
        with st.spinner("Executing Hybrid Retrieval and Traversing Graph..."):
            try:
                resp = requests.post(
                    f"{api_url}/api/search",
                    json={"query": query_input, "language": lang_code, "history": []},
                    timeout=20
                )
                if resp.status_code == 200:
                    data = resp.json()

                    # Display Groundedness Badge
                    if data.get("is_grounded"):
                        st.success("✅ **Verified Graph Grounding**: Results verified against RxNorm and OpenFDA ontological links.")
                    else:
                        st.warning("⚠️ **Zero Graph Matches**: Model refused speculation pursuant to medical guardrails.")

                    # Clinical Summary Card
                    st.markdown(f"""
                    <div class="clinical-card">
                        <h3>📋 Clinical Consultation Summary</h3>
                        {data.get("answer", "")}
                    </div>
                    """, unsafe_allow_html=True)

                    # Metrics Overview
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Symptoms Identified", len(data.get("symptoms_identified", [])))
                        for s in data.get("symptoms_identified", []):
                            st.markdown(f"<span class='badge-symptom'>🔴 {s}</span> ", unsafe_allow_html=True)
                    with c2:
                        st.metric("Active Compositions", len(data.get("compositions", [])))
                        for comp in data.get("compositions", []):
                            st.markdown(f"<span class='badge-ing'>🔵 {comp}</span> ", unsafe_allow_html=True)
                    with c3:
                        st.metric("Alternative Brands Found", len(data.get("alternative_medicines", [])))
                        st.write(", ".join(data.get("alternative_medicines", [])[:6]) or "None")

                    # Graph Visualization
                    st.subheader("🕸️ Graph Traversal Subgraph")
                    render_network_graph(data.get("graph", {}))

                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as ex:
                st.error(f"Failed to connect to backend service at {api_url}: {ex}")


# ---------------------------------------------------------------------------
# TAB 2: Direct Brand Alternatives Lookup
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("💊 Look Up Alternative Brands for a Medication")
    target_brand = st.text_input("Enter Medication Brand Name:", value="Lipitor")
    if st.button("Find Alternatives", type="primary"):
        with st.spinner(f"Traversing graph for {target_brand}..."):
            try:
                res = requests.get(f"{api_url}/api/alternatives/{target_brand}", timeout=15)
                if res.status_code == 200:
                    alt_data = res.json()
                    st.success(f"Found active composition: **{alt_data.get('active_ingredient')}**")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Target Brand**: `{alt_data.get('queried_brand')}`")
                        st.markdown(f"**Active Composition (RxNorm)**: `{alt_data.get('active_ingredient')}`")
                        if alt_data.get("rxcui"):
                            st.markdown(f"**RXCUI**: `{alt_data.get('rxcui')}`")

                    with col_b:
                        st.markdown("**Equivalent Alternative Brands:**")
                        for b in alt_data.get("alternative_brands", []):
                            st.markdown(f"- 🟢 {b}")

                    st.subheader("Sub-Graph Visualization")
                    render_network_graph(alt_data.get("graph", {}))
                else:
                    st.error(f"Brand not found: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.divider()
st.caption("⚖️ **Medical Safety Disclaimer**: MedGraph Nexus is an algorithmic pharmaceutical reference. Always consult a licensed physician before taking any medication.")
