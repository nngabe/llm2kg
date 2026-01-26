#!/usr/bin/env python3
"""
Tests for wikidata_kg_builder.py

Tests cover:
- SPARQL query construction
- SPARQL client execution
- WikiPage Neo4j loader operations
- KG Builder BFS logic
- Integration tests
"""

import os
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import sys
sys.path.insert(0, '/app')

from wikidata_kg_builder import (
    WikidataEntity,
    WikidataRelationship,
    WikidataFetchResult,
    WikidataSPARQLClient,
    WikiPageLoader,
    FalkorDBPageLoader,
    WikidataKGBuilder,
    PROPERTY_SUBCLASS_OF,
    PROPERTY_INSTANCE_OF,
    PROPERTY_PART_OF,
    PROPERTY_TO_REL_TYPE,
)


class TestDataModels(unittest.TestCase):
    """Tests for data model classes."""

    def test_wikidata_entity_creation(self):
        """Test WikidataEntity creation with all fields."""
        entity = WikidataEntity(
            qid="Q1101",
            name="Technology",
            wikipedia_url="https://en.wikipedia.org/wiki/Technology",
            description="application of knowledge for practical goals",
        )
        self.assertEqual(entity.qid, "Q1101")
        self.assertEqual(entity.name, "Technology")
        self.assertIsNotNone(entity.wikipedia_url)
        self.assertIsNotNone(entity.description)

    def test_wikidata_entity_hash(self):
        """Test WikidataEntity hash and equality."""
        e1 = WikidataEntity(qid="Q1101", name="Technology")
        e2 = WikidataEntity(qid="Q1101", name="Tech")  # Same QID, different name
        e3 = WikidataEntity(qid="Q21198", name="Computer science")

        self.assertEqual(hash(e1), hash(e2))
        self.assertEqual(e1, e2)
        self.assertNotEqual(e1, e3)

    def test_wikidata_relationship_creation(self):
        """Test WikidataRelationship creation."""
        rel = WikidataRelationship(
            source_qid="Q21198",
            target_qid="Q1101",
            property_id="P279",
            property_label="subclass_of",
        )
        self.assertEqual(rel.source_qid, "Q21198")
        self.assertEqual(rel.target_qid, "Q1101")
        self.assertEqual(rel.property_id, "P279")

    def test_wikidata_relationship_hash(self):
        """Test WikidataRelationship hash and equality."""
        r1 = WikidataRelationship("Q1", "Q2", "P279", "subclass_of")
        r2 = WikidataRelationship("Q1", "Q2", "P279", "subclass")  # Same triple
        r3 = WikidataRelationship("Q1", "Q2", "P31", "instance_of")

        self.assertEqual(hash(r1), hash(r2))
        self.assertEqual(r1, r2)
        self.assertNotEqual(r1, r3)


class TestSPARQLQueries(unittest.TestCase):
    """Tests for SPARQL query construction."""

    def setUp(self):
        self.client = WikidataSPARQLClient()

    def test_combined_query_construction_single_parent(self):
        """Test SPARQL query construction with a single parent."""
        query = self.client._build_combined_query(
            parent_qids=["Q1101"],
            include_subclasses=True,
            include_instances=True,
            include_parts=True,
            limit=100,
        )

        # Check query contains expected elements
        self.assertIn("wd:Q1101", query)
        self.assertIn("wdt:P279", query)  # subclass_of
        self.assertIn("wdt:P31", query)   # instance_of
        self.assertIn("wdt:P361", query)  # part_of
        self.assertIn("LIMIT 100", query)

    def test_combined_query_construction_multiple_parents(self):
        """Test SPARQL query with multiple parent QIDs."""
        query = self.client._build_combined_query(
            parent_qids=["Q1101", "Q21198"],
            include_subclasses=True,
            include_instances=False,
            include_parts=False,
            limit=500,
        )

        # Check both QIDs in VALUES clause
        self.assertIn("wd:Q1101", query)
        self.assertIn("wd:Q21198", query)
        # Only subclass should be included
        self.assertIn("wdt:P279", query)
        self.assertNotIn("wdt:P31", query)
        self.assertNotIn("wdt:P361", query)
        self.assertIn("LIMIT 500", query)

    def test_query_requires_at_least_one_relationship_type(self):
        """Test that query construction fails without any relationship types."""
        with self.assertRaises(ValueError):
            self.client._build_combined_query(
                parent_qids=["Q1101"],
                include_subclasses=False,
                include_instances=False,
                include_parts=False,
            )


class TestSPARQLClient(unittest.TestCase):
    """Tests for SPARQL client execution (requires network)."""

    @classmethod
    def setUpClass(cls):
        """Skip tests if no network or Wikidata is unreachable."""
        cls.client = WikidataSPARQLClient(rate_limit_delay=0.5)
        try:
            # Quick check if Wikidata is reachable
            cls.client.fetch_entity("Q1101")
            cls.can_run = True
        except Exception:
            cls.can_run = False

    def test_fetch_technology_entity(self):
        """Test fetching the Technology entity (Q1101)."""
        if not self.can_run:
            self.skipTest("Wikidata not reachable")

        entity = self.client.fetch_entity("Q1101")
        self.assertIsNotNone(entity)
        self.assertEqual(entity.qid, "Q1101")
        self.assertIn("technology", entity.name.lower())

    def test_fetch_technology_subclasses(self):
        """Test fetching subclasses of Technology."""
        if not self.can_run:
            self.skipTest("Wikidata not reachable")

        result = self.client.fetch_children(
            parent_qids=["Q1101"],
            limit=10,
            include_subclasses=True,
            include_instances=False,
            include_parts=False,
        )

        self.assertIsInstance(result, WikidataFetchResult)
        self.assertGreater(len(result.entities), 0)
        self.assertGreater(len(result.relationships), 0)

        # All relationships should be subclass_of
        for rel in result.relationships:
            self.assertEqual(rel.property_id, "P279")

    def test_rate_limiting(self):
        """Test that rate limiting is applied."""
        if not self.can_run:
            self.skipTest("Wikidata not reachable")

        client = WikidataSPARQLClient(rate_limit_delay=1.0)

        start = time.time()
        client.fetch_entity("Q1101")
        client.fetch_entity("Q21198")
        elapsed = time.time() - start

        # Should have waited at least 1 second between requests
        self.assertGreaterEqual(elapsed, 1.0)


class TestWikiPageLoader(unittest.TestCase):
    """Tests for WikiPage Neo4j loader (requires Neo4j)."""

    @classmethod
    def setUpClass(cls):
        """Set up Neo4j connection for tests."""
        try:
            cls.loader = WikiPageLoader()
            # Test connection
            cls.loader.get_stats()
            cls.can_run = True
        except Exception:
            cls.can_run = False
            cls.loader = None

    @classmethod
    def tearDownClass(cls):
        """Close Neo4j connection."""
        if cls.loader:
            cls.loader.close()

    def test_init_schema(self):
        """Test schema initialization."""
        if not self.can_run:
            self.skipTest("Neo4j not available")

        # Should not raise
        self.loader.init_schema()

    def test_batch_create_entities(self):
        """Test creating WikiPage entities."""
        if not self.can_run:
            self.skipTest("Neo4j not available")

        # Create test entities
        entities = [
            WikidataEntity(qid="TEST_Q1", name="Test Entity 1"),
            WikidataEntity(qid="TEST_Q2", name="Test Entity 2"),
        ]

        count = self.loader.batch_create_entities(entities)
        self.assertEqual(count, 2)

        # Clean up
        with self.loader.driver.session() as session:
            session.run("MATCH (w:WikiPage) WHERE w.wikidata_id STARTS WITH 'TEST_' DELETE w")

    def test_batch_create_relationships(self):
        """Test creating relationships between WikiPage entities."""
        if not self.can_run:
            self.skipTest("Neo4j not available")

        # First create test entities
        entities = [
            WikidataEntity(qid="TEST_REL_Q1", name="Parent Entity"),
            WikidataEntity(qid="TEST_REL_Q2", name="Child Entity"),
        ]
        self.loader.batch_create_entities(entities)

        # Create relationship
        relationships = [
            WikidataRelationship(
                source_qid="TEST_REL_Q2",
                target_qid="TEST_REL_Q1",
                property_id="P279",
                property_label="subclass_of",
            )
        ]

        count = self.loader.batch_create_relationships(relationships)
        self.assertEqual(count, 1)

        # Clean up
        with self.loader.driver.session() as session:
            session.run("""
                MATCH (w:WikiPage)
                WHERE w.wikidata_id STARTS WITH 'TEST_REL_'
                DETACH DELETE w
            """)

    def test_idempotent_merge(self):
        """Test that entity creation is idempotent (MERGE behavior)."""
        if not self.can_run:
            self.skipTest("Neo4j not available")

        entity = WikidataEntity(qid="TEST_IDEM_Q1", name="Idempotent Test")

        # Create twice
        self.loader.batch_create_entities([entity])
        self.loader.batch_create_entities([entity])

        # Should only have one node
        with self.loader.driver.session() as session:
            result = session.run("""
                MATCH (w:WikiPage {wikidata_id: 'TEST_IDEM_Q1'})
                RETURN count(w) as count
            """)
            count = result.single()["count"]
            self.assertEqual(count, 1)

            # Clean up
            session.run("MATCH (w:WikiPage {wikidata_id: 'TEST_IDEM_Q1'}) DELETE w")

    def test_search_by_name(self):
        """Test searching WikiPage nodes by name."""
        if not self.can_run:
            self.skipTest("Neo4j not available")

        # Create test entity
        entity = WikidataEntity(
            qid="TEST_SEARCH_Q1",
            name="Machine Learning Test",
            wikipedia_url="https://example.com/ml",
        )
        self.loader.batch_create_entities([entity])

        # Search
        results = self.loader.search_by_name("machine learning", limit=5)

        # Should find our test entity
        found = any(r["qid"] == "TEST_SEARCH_Q1" for r in results)
        self.assertTrue(found)

        # Clean up
        with self.loader.driver.session() as session:
            session.run("MATCH (w:WikiPage {wikidata_id: 'TEST_SEARCH_Q1'}) DELETE w")


class TestKGBuilder(unittest.TestCase):
    """Tests for KGBuilder BFS logic."""

    def test_build_dry_run(self):
        """Test building KG in dry-run mode (no Neo4j writes)."""
        # Mock the SPARQL client
        mock_client = Mock(spec=WikidataSPARQLClient)

        # Mock fetch_entity for root
        mock_client.fetch_entity.return_value = WikidataEntity(
            qid="Q1101",
            name="Technology",
        )

        # Mock fetch_children for level 1
        mock_client.fetch_children.return_value = WikidataFetchResult(
            entities=[
                WikidataEntity(qid="Q21198", name="Computer science"),
                WikidataEntity(qid="Q11016", name="Engineering"),
            ],
            relationships=[
                WikidataRelationship("Q21198", "Q1101", "P279", "subclass_of"),
                WikidataRelationship("Q11016", "Q1101", "P279", "subclass_of"),
            ],
        )

        builder = WikidataKGBuilder(sparql_client=mock_client, dry_run=True)
        stats = builder.build_from_root(
            root_qid="Q1101",
            max_depth=1,
            max_entities_per_level=10,
        )

        self.assertTrue(stats.get("dry_run"))
        self.assertEqual(stats["total_visited"], 3)  # root + 2 children
        self.assertEqual(stats.get("entities_found"), 3)
        self.assertEqual(stats.get("relationships_found"), 2)

    def test_visited_tracking_no_duplicates(self):
        """Test that visited tracking prevents duplicates."""
        mock_client = Mock(spec=WikidataSPARQLClient)

        # Mock fetch_entity
        mock_client.fetch_entity.return_value = WikidataEntity(qid="Q1", name="Root")

        # Level 1: Return Q2, Q3
        # Level 2: Return Q3 again (should be filtered), Q4
        call_count = [0]

        def mock_fetch_children(parent_qids, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return WikidataFetchResult(
                    entities=[
                        WikidataEntity(qid="Q2", name="Entity 2"),
                        WikidataEntity(qid="Q3", name="Entity 3"),
                    ],
                    relationships=[
                        WikidataRelationship("Q2", "Q1", "P279", "subclass_of"),
                        WikidataRelationship("Q3", "Q1", "P279", "subclass_of"),
                    ],
                )
            else:
                # Second call returns Q3 again (duplicate) and Q4 (new)
                return WikidataFetchResult(
                    entities=[
                        WikidataEntity(qid="Q3", name="Entity 3 Duplicate"),
                        WikidataEntity(qid="Q4", name="Entity 4"),
                    ],
                    relationships=[
                        WikidataRelationship("Q3", "Q2", "P279", "subclass_of"),
                        WikidataRelationship("Q4", "Q2", "P279", "subclass_of"),
                    ],
                )

        mock_client.fetch_children.side_effect = mock_fetch_children

        builder = WikidataKGBuilder(sparql_client=mock_client, dry_run=True)
        stats = builder.build_from_root(
            root_qid="Q1",
            max_depth=2,
        )

        # Should visit Q1, Q2, Q3, Q4 - not Q3 twice
        self.assertEqual(stats["total_visited"], 4)


class TestIntegration(unittest.TestCase):
    """Integration tests (require both Wikidata and Neo4j)."""

    @classmethod
    def setUpClass(cls):
        """Check if integration tests can run."""
        cls.can_run_wikidata = False
        cls.can_run_neo4j = False

        try:
            client = WikidataSPARQLClient(rate_limit_delay=0.5)
            client.fetch_entity("Q1101")
            cls.can_run_wikidata = True
        except Exception:
            pass

        try:
            loader = WikiPageLoader()
            loader.get_stats()
            loader.close()
            cls.can_run_neo4j = True
        except Exception:
            pass

    def test_small_build_integration(self):
        """Test building a small KG (depth=1, limit=5)."""
        if not self.can_run_wikidata or not self.can_run_neo4j:
            self.skipTest("Wikidata or Neo4j not available")

        builder = WikidataKGBuilder()
        try:
            # Use a well-known specific entity for testing
            stats = builder.build_from_root(
                root_qid="Q21198",  # Computer science
                max_depth=1,
                max_entities_per_level=5,
                include_instances=False,
                include_parts=False,
            )

            self.assertGreater(stats["total_visited"], 1)
            self.assertIn("entities_created", stats)

        finally:
            builder.close()

    def test_wiki_search_finds_local_pages(self):
        """Test that wiki_search can find pages from local WikiPage nodes."""
        if not self.can_run_neo4j:
            self.skipTest("Neo4j not available")

        loader = WikiPageLoader()
        try:
            # Create a test page
            entity = WikidataEntity(
                qid="TEST_WIKI_SEARCH_Q1",
                name="Machine Learning Integration Test",
                wikipedia_url="https://en.wikipedia.org/wiki/Machine_learning",
            )
            loader.batch_create_entities([entity])

            # Search should find it
            results = loader.search_by_name("Machine Learning", limit=10)
            found = any("TEST_WIKI_SEARCH" in r["qid"] for r in results)
            self.assertTrue(found)

        finally:
            # Clean up
            with loader.driver.session() as session:
                session.run("MATCH (w:WikiPage) WHERE w.wikidata_id STARTS WITH 'TEST_WIKI_SEARCH' DELETE w")
            loader.close()


class TestFalkorDBPageLoader(unittest.TestCase):
    """Tests for FalkorDB WikiPage loader (requires FalkorDB)."""

    @classmethod
    def setUpClass(cls):
        """Set up FalkorDB connection for tests."""
        try:
            cls.loader = FalkorDBPageLoader()
            # Test connection
            cls.loader.get_stats()
            cls.can_run = True
        except Exception as e:
            print(f"FalkorDB not available: {e}")
            cls.can_run = False
            cls.loader = None

    @classmethod
    def tearDownClass(cls):
        """Close FalkorDB connection."""
        if cls.loader:
            cls.loader.close()

    def test_init_schema(self):
        """Test schema initialization."""
        if not self.can_run:
            self.skipTest("FalkorDB not available")

        # Should not raise
        self.loader.init_schema()

    def test_batch_create_entities(self):
        """Test creating WikiPage entities in FalkorDB."""
        if not self.can_run:
            self.skipTest("FalkorDB not available")

        # Create test entities
        entities = [
            WikidataEntity(qid="FTEST_Q1", name="FalkorDB Test Entity 1"),
            WikidataEntity(qid="FTEST_Q2", name="FalkorDB Test Entity 2"),
        ]

        count = self.loader.batch_create_entities(entities)
        self.assertEqual(count, 2)

        # Clean up
        try:
            self.loader.graph.query("MATCH (w:WikiPage) WHERE w.wikidata_id STARTS WITH 'FTEST_' DELETE w")
        except Exception:
            pass

    def test_batch_create_relationships(self):
        """Test creating relationships in FalkorDB."""
        if not self.can_run:
            self.skipTest("FalkorDB not available")

        # First create test entities
        entities = [
            WikidataEntity(qid="FTEST_REL_Q1", name="FalkorDB Parent"),
            WikidataEntity(qid="FTEST_REL_Q2", name="FalkorDB Child"),
        ]
        self.loader.batch_create_entities(entities)

        # Create relationship
        relationships = [
            WikidataRelationship(
                source_qid="FTEST_REL_Q2",
                target_qid="FTEST_REL_Q1",
                property_id="P279",
                property_label="subclass_of",
            )
        ]

        count = self.loader.batch_create_relationships(relationships)
        self.assertEqual(count, 1)

        # Clean up
        try:
            self.loader.graph.query("""
                MATCH (w:WikiPage)
                WHERE w.wikidata_id STARTS WITH 'FTEST_REL_'
                DETACH DELETE w
            """)
        except Exception:
            pass

    def test_search_by_name(self):
        """Test searching WikiPage nodes by name in FalkorDB."""
        if not self.can_run:
            self.skipTest("FalkorDB not available")

        # Create test entity
        entity = WikidataEntity(
            qid="FTEST_SEARCH_Q1",
            name="FalkorDB Search Test Entity",
            wikipedia_url="https://example.com/falkor",
        )
        self.loader.batch_create_entities([entity])

        # Search
        results = self.loader.search_by_name("FalkorDB Search", limit=5)

        # Should find our test entity
        found = any("FTEST_SEARCH" in r["qid"] for r in results)
        self.assertTrue(found)

        # Clean up
        try:
            self.loader.graph.query("MATCH (w:WikiPage {wikidata_id: 'FTEST_SEARCH_Q1'}) DELETE w")
        except Exception:
            pass


class TestBackendSelection(unittest.TestCase):
    """Tests for backend selection in WikidataKGBuilder."""

    def test_neo4j_backend_selection(self):
        """Test that neo4j backend is selected correctly."""
        builder = WikidataKGBuilder(backend="neo4j", dry_run=True)
        self.assertEqual(builder.backend, "neo4j")
        builder.close()

    def test_falkordb_backend_selection(self):
        """Test that falkordb backend is selected correctly in dry run mode."""
        builder = WikidataKGBuilder(backend="falkordb", dry_run=True)
        self.assertEqual(builder.backend, "falkordb")
        builder.close()


class TestPropertyMapping(unittest.TestCase):
    """Tests for property-to-relationship-type mapping."""

    def test_property_to_rel_type_mapping(self):
        """Test that all properties map to correct relationship types."""
        self.assertEqual(PROPERTY_TO_REL_TYPE[PROPERTY_SUBCLASS_OF], "SUBCLASS_OF")
        self.assertEqual(PROPERTY_TO_REL_TYPE[PROPERTY_INSTANCE_OF], "INSTANCE_OF")
        self.assertEqual(PROPERTY_TO_REL_TYPE[PROPERTY_PART_OF], "PART_OF")


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataModels))
    suite.addTests(loader.loadTestsFromTestCase(TestSPARQLQueries))
    suite.addTests(loader.loadTestsFromTestCase(TestSPARQLClient))
    suite.addTests(loader.loadTestsFromTestCase(TestWikiPageLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestFalkorDBPageLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestBackendSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestKGBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPropertyMapping))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
