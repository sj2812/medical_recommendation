# src/kg_embed_gds.py
"""
Compute node embeddings on the KG using Neo4j Graph Data Science (GDS).
Requires the GDS plugin to be installed in your Neo4j instance.
"""
import os
import sys

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "qbankuser")

GRAPH_NAME = os.environ.get("GDS_GRAPH_NAME", "qbankGraph")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "64"))
WRITE_PROP = os.environ.get("GDS_WRITE_PROP", "embedding")


def check_gds(session):
    """Verify GDS is installed; raise SystemExit with instructions if not."""
    try:
        # GDS 2.x uses function: RETURN gds.version(); older GDS used CALL gds.version() YIELD version
        result = session.run("RETURN gds.version() AS version").single()
        if result and result["version"]:
            print("GDS version:", result["version"])
            return True
        raise ClientError("", "GDS version returned empty")
    except ClientError as e:
        if "ProcedureNotFound" in str(e) or "gds.version" in str(e) or "Procedure" in str(e):
            print("Neo4j Graph Data Science (GDS) is not available.", file=sys.stderr)
            print("Neo4j error:", e, file=sys.stderr)
            print(
                "If you just installed GDS: fully STOP and START Neo4j (or the database in Desktop).\n"
                "Then in Neo4j Browser run: CALL gds.version() to confirm.\n"
                "  - Neo4j Desktop: Plugins → Graph Data Science → Install, then restart the DB\n"
                "  - Docker: add the GDS plugin JAR to the plugins folder and restart the container",
                file=sys.stderr,
            )
            sys.exit(1)
        raise


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session(database=NEO4J_DATABASE) as session:
        check_gds(session)

        # Drop projected graph if it exists (no APOC required)
        result = session.run("CALL gds.graph.exists($name) YIELD exists RETURN exists", name=GRAPH_NAME).single()
        if result and result["exists"]:
            session.run("CALL gds.graph.drop($name)", name=GRAPH_NAME)
            print("Dropped existing graph:", GRAPH_NAME)

        # Project graph (undirected relationships)
        # We include Question-Term and Term-Category. Difficulty is optional.
        session.run(
            """
            CALL gds.graph.project(
              $name,
              ['Question','Term','Category'],
              {
                COVERS_TERM: {type: 'COVERS_TERM', orientation: 'UNDIRECTED'},
                PART_OF_CATEGORY: {type: 'PART_OF_CATEGORY', orientation: 'UNDIRECTED'},
                BELONGS_TO_CATEGORY: {type: 'BELONGS_TO_CATEGORY', orientation: 'UNDIRECTED'}
              }
            )
            """,
            name=GRAPH_NAME,
        )
        print("Projected graph:", GRAPH_NAME)

        # Compute embeddings
        session.run(
            """
            CALL gds.fastRP.write($name, {
              embeddingDimension: $dim,
              randomSeed: 42,
              iterationWeights: [0.8, 1.0, 1.0],
              writeProperty: $prop
            })
            """,
            name=GRAPH_NAME,
            dim=EMBED_DIM,
            prop=WRITE_PROP,
        )
        print(f"Wrote FastRP embeddings to :Question.{WRITE_PROP} (dim={EMBED_DIM})")

    driver.close()


if __name__ == "__main__":
    main()
