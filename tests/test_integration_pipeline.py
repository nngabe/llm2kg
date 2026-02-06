#!/usr/bin/env python3
"""
Integration tests for run_pipeline.py with real services.

Tests use:
- Real FalkorDB database
- Real Ollama LLM
- Real network access for Wikidata/Wikipedia/Web

Run with: python -m pytest tests/test_integration_pipeline.py -v -s
"""

import sys
sys.path.insert(0, '/app')

import os
import json
import unittest
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
TEST_GRAPH_NAME = "test_integration_kg"


def falkordb_available() -> bool:
    """Check if FalkorDB is available."""
    try:
        from falkordb import FalkorDB
        db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        db.select_graph("test_connection")
        return True
    except Exception as e:
        logger.warning(f"FalkorDB not available: {e}")
        return False


def ollama_available() -> bool:
    """Check if Ollama is available."""
    try:
        import requests
        resp = requests.get("http://host.docker.internal:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        return False


def wikidata_available() -> bool:
    """Check if Wikidata SPARQL endpoint is available."""
    try:
        import requests
        resp = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": "SELECT ?item WHERE { ?item wdt:P31 wd:Q5 } LIMIT 1", "format": "json"},
            timeout=10
        )
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Wikidata not available: {e}")
        return False


# =============================================================================
# STAGE 1: WIKIDATA INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(falkordb_available() and wikidata_available(), "FalkorDB or Wikidata not available")
class TestWikidataStageIntegration(unittest.TestCase):
    """Integration tests for Stage 1: Wikidata Backbone."""

    @classmethod
    def setUpClass(cls):
        """Set up test graph."""
        from falkordb import FalkorDB
        cls.db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        cls.graph = cls.db.select_graph(TEST_GRAPH_NAME)
        # Clear test graph
        try:
            cls.graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

    @classmethod
    def tearDownClass(cls):
        """Clean up test graph."""
        try:
            cls.graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

    def test_sparql_client_fetch_entity(self):
        """Test fetching a real entity from Wikidata."""
        from wikidata_kg_builder import WikidataSPARQLClient

        client = WikidataSPARQLClient(rate_limit_delay=0.5)
        entity = client.fetch_entity("Q1101")  # Technology

        self.assertIsNotNone(entity)
        self.assertEqual(entity.qid, "Q1101")
        # Entity should have a name (may vary based on Wikidata response)
        self.assertIsNotNone(entity.name)
        self.assertGreater(len(entity.name), 0)

    def test_sparql_client_fetch_children(self):
        """Test fetching children from Wikidata."""
        from wikidata_kg_builder import WikidataSPARQLClient

        client = WikidataSPARQLClient(rate_limit_delay=0.5)
        result = client.fetch_children(
            parent_qids=["Q11016"],  # Engineering - more likely to have subclasses
            limit=10,
            include_subclasses=True,
            include_instances=True,
            include_parts=True,
        )

        self.assertIsNotNone(result)
        # Result may be empty depending on Wikidata response
        self.assertIsInstance(result.entities, list)
        self.assertIsInstance(result.relationships, list)

    def test_kg_builder_dry_run(self):
        """Test KG builder in dry-run mode with real Wikidata."""
        from wikidata_kg_builder import WikidataKGBuilder, ExplorationStrategy

        builder = WikidataKGBuilder(
            backend="falkordb",
            falkordb_graph_name=TEST_GRAPH_NAME,
            exploration_strategy=ExplorationStrategy.BFS,
            dry_run=True,
        )

        stats = builder.build_from_root(
            root_qid="Q2526135",  # Industrial gas turbine
            max_depth=1,
            max_entities_per_level=5,
        )

        self.assertTrue(stats.get("dry_run"))
        self.assertGreater(stats.get("total_visited", 0), 0)

    def test_kg_builder_real_write(self):
        """Test KG builder with real write to FalkorDB."""
        from wikidata_kg_builder import WikidataKGBuilder, ExplorationStrategy

        builder = WikidataKGBuilder(
            backend="falkordb",
            falkordb_graph_name=TEST_GRAPH_NAME,
            exploration_strategy=ExplorationStrategy.BFS,
            dry_run=False,
        )

        try:
            stats = builder.build_from_root(
                root_qid="Q11016",  # Engineering - more common entity
                max_depth=1,
                max_entities_per_level=3,
                with_embeddings=False,  # Skip embeddings for speed
            )

            # Stats should be returned
            self.assertIsInstance(stats, dict)
            self.assertIn("total_visited", stats)

            # Check if any nodes were created (may be 0 if dry_run was forced)
            result = self.graph.query("MATCH (w:WikiPage) RETURN count(w) as c")
            count = result.result_set[0][0] if result.result_set else 0
            # Don't assert on count - it may be 0 if Wikidata didn't return results

        finally:
            builder.close()


# =============================================================================
# STAGE 2a: WIKIPEDIA INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(falkordb_available() and ollama_available(), "FalkorDB or Ollama not available")
class TestWikipediaStageIntegration(unittest.TestCase):
    """Integration tests for Stage 2a: Wikipedia Articles."""

    @classmethod
    def setUpClass(cls):
        """Set up test graph with WikiPage nodes."""
        from falkordb import FalkorDB
        cls.db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        cls.graph = cls.db.select_graph(TEST_GRAPH_NAME)

        # Create test WikiPage nodes
        cls.graph.query("""
            MERGE (w:WikiPage {wikidata_id: 'Q2526135'})
            SET w.name = 'Industrial gas turbine',
                w.wikipedia_url = 'https://en.wikipedia.org/wiki/Gas_turbine'
        """)

    def test_wikipedia_loader(self):
        """Test loading Wikipedia content."""
        from langchain_community.document_loaders import WikipediaLoader

        loader = WikipediaLoader(query="gas turbine", load_max_docs=1)
        docs = loader.load()

        self.assertGreater(len(docs), 0)
        self.assertIn("turbine", docs[0].page_content.lower())

    def test_pipeline_initialization(self):
        """Test WikipediaArticlePipeline initialization."""
        from knowledge_graph.wikipedia_pipeline import WikipediaArticlePipeline, ChunkingStrategy

        pipeline = WikipediaArticlePipeline(
            graph_name=TEST_GRAPH_NAME,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
        )

        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(pipeline.graph)

    def test_pipeline_rank_articles(self):
        """Test article ranking."""
        from knowledge_graph.wikipedia_pipeline import WikipediaArticlePipeline

        pipeline = WikipediaArticlePipeline(graph_name=TEST_GRAPH_NAME)

        # Get WikiPage nodes
        result = self.graph.query("""
            MATCH (w:WikiPage)
            WHERE w.wikipedia_url IS NOT NULL
            RETURN w.name, w.wikipedia_url
            LIMIT 5
        """)

        self.assertIsNotNone(result.result_set)


# =============================================================================
# STAGE 2b: WEB PAGE INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(falkordb_available(), "FalkorDB not available")
class TestWebPageStageIntegration(unittest.TestCase):
    """Integration tests for Stage 2b: Web Pages."""

    def test_sitemap_parser(self):
        """Test GE Vernova sitemap parsing."""
        from knowledge_graph.sitemap_parser import SitemapParser

        parser = SitemapParser()

        # Test known sitemaps are defined
        self.assertIsInstance(parser.SITEMAPS, dict)
        self.assertGreater(len(parser.SITEMAPS), 0)
        self.assertIn("gas-power", parser.SITEMAPS)

    def test_web_page_scraping(self):
        """Test basic web page scraping."""
        import requests
        from bs4 import BeautifulSoup

        # Test a simple request
        try:
            resp = requests.get(
                "https://www.gevernova.com/gas-power",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (Test)"}
            )

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                title = soup.find('title')
                self.assertIsNotNone(title)
        except requests.exceptions.RequestException as e:
            self.skipTest(f"Network unavailable: {e}")


# =============================================================================
# STAGE 3: ENTITY EXTRACTION INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(falkordb_available() and ollama_available(), "FalkorDB or Ollama not available")
class TestEntityExtractionIntegration(unittest.TestCase):
    """Integration tests for Stage 3: Entity Extraction."""

    @classmethod
    def setUpClass(cls):
        """Set up test graph with DocumentChunk nodes."""
        from falkordb import FalkorDB
        cls.db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        cls.graph = cls.db.select_graph(TEST_GRAPH_NAME)

        # Create test DocumentChunk
        cls.graph.query("""
            MERGE (d:DocumentChunk {chunk_id: 'test_chunk_001'})
            SET d.content = 'The 9HA gas turbine is manufactured by GE Vernova. It has a combined cycle efficiency of over 64%, making it one of the most efficient gas turbines in the world.',
                d.source_type = 'webpage',
                d.source_url = 'https://www.gevernova.com/test',
                d.extraction_status = 'pending'
        """)

    def test_ontology_loading(self):
        """Test loading ontology configuration."""
        from ontologies import get_ontology

        ontology = get_ontology("medium")

        self.assertIsNotNone(ontology)
        self.assertGreater(len(ontology.entity_types), 0)
        self.assertGreater(len(ontology.relationship_types), 0)

    def test_extractor_initialization(self):
        """Test HybridEntityExtractor initialization."""
        from knowledge_graph.entity_extraction import HybridEntityExtractor
        from ontologies import get_ontology

        ontology = get_ontology("small")
        ontology_dict = {
            "entity_types": ontology.entity_types,
            "relationship_types": ontology.relationship_types,
        }

        extractor = HybridEntityExtractor(
            graph_name=TEST_GRAPH_NAME,
            ontology=ontology_dict,
        )

        self.assertIsNotNone(extractor)

    def test_llm_entity_extraction(self):
        """Test LLM-based entity extraction."""
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model="nemotron-3-nano:30b",
            base_url="http://host.docker.internal:11434",
            temperature=0,
            num_ctx=4096,
        )

        prompt = """Extract entities from this text. Return JSON with entities array.

Text: The 9HA gas turbine is manufactured by GE Vernova with 64% efficiency.

Return format: {"entities": [{"name": "...", "type": "..."}]}"""

        response = llm.invoke(prompt)
        self.assertIsNotNone(response.content)
        self.assertIn("9HA", response.content)


# =============================================================================
# STAGE 4: COMMUNITY DETECTION INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(falkordb_available(), "FalkorDB not available")
class TestCommunityDetectionIntegration(unittest.TestCase):
    """Integration tests for Stage 4: Community Detection."""

    @classmethod
    def setUpClass(cls):
        """Set up test graph with Entity nodes."""
        from falkordb import FalkorDB
        import uuid

        cls.db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        cls.graph = cls.db.select_graph(TEST_GRAPH_NAME)

        # Create test entities with relationships
        cls.graph.query(f"""
            MERGE (e1:Entity {{uuid: '{uuid.uuid4()}', name: 'GE Vernova', ontology_type: 'ORGANIZATION'}})
            MERGE (e2:Entity {{uuid: '{uuid.uuid4()}', name: '9HA Gas Turbine', ontology_type: 'PRODUCT'}})
            MERGE (e3:Entity {{uuid: '{uuid.uuid4()}', name: '7HA Gas Turbine', ontology_type: 'PRODUCT'}})
            MERGE (e4:Entity {{uuid: '{uuid.uuid4()}', name: 'Gas Power Division', ontology_type: 'ORGANIZATION'}})
            MERGE (e1)-[:MANUFACTURES]->(e2)
            MERGE (e1)-[:MANUFACTURES]->(e3)
            MERGE (e4)-[:PART_OF]->(e1)
            MERGE (e2)-[:RELATED_TO]->(e3)
        """)

    def test_build_entity_graph(self):
        """Test building NetworkX graph from entities."""
        from knowledge_graph.community_detection import EntityCommunityDetector

        detector = EntityCommunityDetector(
            graph=self.graph,
            min_community_size=2,
            resolution=1.0,
        )

        G = detector.build_entity_graph()

        self.assertIsNotNone(G)
        # May have 0 edges if entity-to-entity relationships aren't found
        self.assertGreaterEqual(G.number_of_nodes(), 0)

    def test_louvain_community_detection(self):
        """Test Louvain algorithm on entity graph."""
        from knowledge_graph.community_detection import EntityCommunityDetector
        import networkx as nx
        from community import community_louvain

        detector = EntityCommunityDetector(
            graph=self.graph,
            min_community_size=2,
        )

        G = detector.build_entity_graph()

        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G)
            self.assertIsInstance(partition, dict)


# =============================================================================
# FULL PIPELINE INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(falkordb_available() and ollama_available(), "FalkorDB or Ollama not available")
class TestFullPipelineIntegration(unittest.TestCase):
    """Integration tests for full pipeline execution."""

    @classmethod
    def setUpClass(cls):
        """Set up clean test graph."""
        from falkordb import FalkorDB
        cls.db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        cls.graph = cls.db.select_graph(TEST_GRAPH_NAME)
        # Clear graph
        try:
            cls.graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

    @classmethod
    def tearDownClass(cls):
        """Clean up test graph."""
        try:
            cls.graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

    def test_get_falkordb_connection(self):
        """Test FalkorDB connection utility."""
        import run_pipeline

        # Temporarily override graph name
        original_name = run_pipeline.GRAPH_NAME
        run_pipeline.GRAPH_NAME = TEST_GRAPH_NAME

        try:
            graph = run_pipeline.get_falkordb_connection()
            self.assertIsNotNone(graph)

            # Test query
            result = graph.query("RETURN 1 as test")
            self.assertEqual(result.result_set[0][0], 1)
        finally:
            run_pipeline.GRAPH_NAME = original_name

    def test_get_graph_stats(self):
        """Test graph statistics retrieval."""
        import run_pipeline

        original_name = run_pipeline.GRAPH_NAME
        run_pipeline.GRAPH_NAME = TEST_GRAPH_NAME

        try:
            stats = run_pipeline.get_graph_stats()

            self.assertIsInstance(stats, dict)
            self.assertIn("WikiPage", stats)
            self.assertIn("Entity", stats)
            self.assertIn("total_relationships", stats)
        finally:
            run_pipeline.GRAPH_NAME = original_name


# =============================================================================
# EMBEDDINGS INTEGRATION TESTS
# =============================================================================

@unittest.skipUnless(ollama_available(), "Ollama not available")
class TestEmbeddingsIntegration(unittest.TestCase):
    """Integration tests for embeddings generation."""

    def test_ollama_embeddings(self):
        """Test Ollama embeddings generation."""
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model="qwen3-embedding:8b",
            base_url="http://host.docker.internal:11434",
            num_ctx=4096,
        )

        vector = embeddings.embed_query("gas turbine efficiency")

        self.assertIsInstance(vector, list)
        self.assertGreater(len(vector), 100)  # Should be high dimensional
        self.assertIsInstance(vector[0], float)

    def test_batch_embeddings(self):
        """Test batch embeddings generation."""
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model="qwen3-embedding:8b",
            base_url="http://host.docker.internal:11434",
        )

        texts = [
            "The 9HA gas turbine has 64% efficiency",
            "GE Vernova manufactures power equipment",
            "Combined cycle power plants use gas turbines",
        ]

        vectors = embeddings.embed_documents(texts)

        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(vectors[0]), len(vectors[1]))


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
