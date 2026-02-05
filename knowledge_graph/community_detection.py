"""
Entity-level community detection using Louvain algorithm.

Implements Zep's community layer architecture:
- Detects communities of semantically related entities
- Uses Louvain modularity optimization on entity relationship graph
- Creates Community nodes and BELONGS_TO relationships

Dependencies:
    pip install python-louvain networkx

Usage:
    from knowledge_graph.community_detection import EntityCommunityDetector

    detector = EntityCommunityDetector(graph, min_community_size=3)
    count = detector.run()
    print(f"Created {count} entity communities")

    # Or with LLM-generated names/summaries
    count = detector.run(generate_summaries=True)
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from .models import Community

logger = logging.getLogger(__name__)


class EntityCommunityDetector:
    """
    Detect communities in entity graph using Louvain algorithm.

    The Louvain algorithm optimizes modularity to find community structure.
    It works iteratively:
    1. Assign each node to its own community
    2. Move nodes to neighboring communities if it improves modularity
    3. Aggregate communities into super-nodes
    4. Repeat until no improvement

    Attributes:
        graph: FalkorDB graph connection
        min_size: Minimum community size to keep
        resolution: Louvain resolution parameter (higher = more communities)
    """

    def __init__(
        self,
        graph,
        min_community_size: int = 3,
        resolution: float = 1.0,
        llm=None,
    ):
        """
        Initialize the community detector.

        Args:
            graph: FalkorDB graph connection.
            min_community_size: Minimum entities per community.
            resolution: Louvain resolution (higher = more, smaller communities).
            llm: Optional LLM for generating community names/summaries.
        """
        self.graph = graph
        self.min_size = min_community_size
        self.resolution = resolution
        self.llm = llm

    def build_entity_graph(self):
        """
        Build NetworkX graph from entity relationships.

        Creates an undirected graph where:
        - Nodes are entities (by entity_id)
        - Edges are relationships between entities
        - Edge weights are relationship confidence scores

        Returns:
            NetworkX Graph object.
        """
        import networkx as nx

        # Query all entity-to-entity relationships
        # Exclude internal relationships like SUPERSEDES, EXTRACTED_FROM, MERGED_INTO
        # Note: fact_status filter is optional since not all pipelines set this property
        result = self.graph.query("""
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE type(r) <> 'SUPERSEDES'
              AND type(r) <> 'EXTRACTED_FROM'
              AND type(r) <> 'MERGED_INTO'
              AND (e1.fact_status IS NULL OR e1.fact_status = 'active')
              AND (e2.fact_status IS NULL OR e2.fact_status = 'active')
            RETURN e1.uuid as src,
                   e2.uuid as tgt,
                   type(r) as rel_type,
                   r.confidence as confidence
        """)

        G = nx.Graph()

        for row in result.result_set:
            src, tgt, rel_type, conf = row
            weight = conf if conf else 1.0

            # Add nodes with metadata
            if not G.has_node(src):
                G.add_node(src)
            if not G.has_node(tgt):
                G.add_node(tgt)

            # Add or update edge (sum weights for multiple relationships)
            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] += weight
            else:
                G.add_edge(src, tgt, weight=weight, rel_types=[rel_type])

        logger.info(f"Built entity graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def detect_communities(self) -> List[Dict[str, Any]]:
        """
        Run Louvain algorithm and return community assignments.

        Returns:
            List of community dictionaries with:
            - community_id: Unique ID
            - member_ids: List of entity IDs
            - member_count: Number of members
            - modularity: Overall modularity score
        """
        try:
            import community as community_louvain
        except ImportError:
            logger.error("python-louvain not installed. Run: pip install python-louvain")
            return []

        G = self.build_entity_graph()

        if G.number_of_nodes() < self.min_size:
            logger.info(f"Graph too small ({G.number_of_nodes()} nodes), skipping community detection")
            return []

        # Run Louvain algorithm
        try:
            partition = community_louvain.best_partition(
                G,
                resolution=self.resolution,
                random_state=42,
            )
            modularity = community_louvain.modularity(partition, G)
            logger.info(f"Louvain detected {len(set(partition.values()))} communities (modularity={modularity:.3f})")
        except Exception as e:
            logger.error(f"Louvain algorithm failed: {e}")
            return []

        # Group entities by community
        communities_map = {}
        for entity_id, comm_id in partition.items():
            if comm_id not in communities_map:
                communities_map[comm_id] = []
            communities_map[comm_id].append(entity_id)

        # Filter by minimum size and create community objects
        communities = []
        for comm_id, members in communities_map.items():
            if len(members) >= self.min_size:
                communities.append({
                    "community_id": str(uuid.uuid4())[:12],
                    "louvain_id": comm_id,
                    "member_ids": members,
                    "member_count": len(members),
                    "modularity": modularity,
                })

        logger.info(f"Kept {len(communities)} communities with >= {self.min_size} members")
        return communities

    def get_entity_names(self, entity_ids: List[str]) -> Dict[str, str]:
        """
        Get entity names for a list of entity IDs.

        Args:
            entity_ids: List of entity IDs.

        Returns:
            Dict mapping entity_id to name.
        """
        if not entity_ids:
            return {}

        ids_str = ", ".join([f"'{eid}'" for eid in entity_ids])
        result = self.graph.query(f"""
            MATCH (e:Entity)
            WHERE e.entity_id IN [{ids_str}]
            RETURN e.entity_id, e.name
        """)

        return {row[0]: row[1] for row in result.result_set}

    def generate_community_summary(
        self,
        community: Dict[str, Any],
        entity_names: Dict[str, str],
    ) -> Tuple[str, Optional[str]]:
        """
        Generate name and summary for a community using LLM.

        Args:
            community: Community dictionary.
            entity_names: Dict mapping entity_id to name.

        Returns:
            Tuple of (name, summary).
        """
        # Get member names
        member_names = [entity_names.get(eid, eid) for eid in community["member_ids"][:20]]

        if self.llm is None:
            # Auto-generate name from common terms
            name = f"Community_{community['community_id'][:6]}"
            return name, None

        try:
            # Use LLM to generate name and summary
            prompt = f"""Analyze this group of related entities and provide:
1. A short descriptive name (2-4 words) for this community
2. A one-sentence summary of what ties these entities together

Entities: {', '.join(member_names)}

Respond in this exact format:
Name: [community name]
Summary: [one sentence summary]"""

            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Parse response
            name = f"Community_{community['community_id'][:6]}"
            summary = None

            for line in content.split('\n'):
                if line.startswith('Name:'):
                    name = line.replace('Name:', '').strip()[:50]
                elif line.startswith('Summary:'):
                    summary = line.replace('Summary:', '').strip()[:200]

            return name, summary

        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
            return f"Community_{community['community_id'][:6]}", None

    def save_communities(
        self,
        communities: List[Dict[str, Any]],
        generate_summaries: bool = False,
    ) -> int:
        """
        Save Community nodes and BELONGS_TO relationships.

        Args:
            communities: List of community dictionaries.
            generate_summaries: Whether to generate LLM summaries.

        Returns:
            Number of communities saved.
        """
        ts = datetime.now(timezone.utc).isoformat()
        saved = 0

        for comm in communities:
            cid = comm["community_id"]

            # Get entity names for summary generation
            entity_names = {}
            if generate_summaries:
                entity_names = self.get_entity_names(comm["member_ids"][:20])
                name, summary = self.generate_community_summary(comm, entity_names)
            else:
                name = f"Community_{cid[:6]}"
                summary = None

            # Get representative entities (top by degree in the community)
            representative = comm["member_ids"][:5]

            try:
                # Create Community node
                summary_str = f"'{summary.replace(chr(39), chr(92) + chr(39))}'" if summary else "null"
                name_escaped = name.replace("'", "\\'")

                self.graph.query(f"""
                    CREATE (c:Community {{
                        community_id: '{cid}',
                        name: '{name_escaped}',
                        summary: {summary_str},
                        algorithm: 'louvain',
                        modularity_score: {comm['modularity']},
                        member_count: {comm['member_count']},
                        representative_entities: {representative},
                        created_at: '{ts}',
                        valid_from: '{ts}'
                    }})
                """)

                # Create BELONGS_TO relationships
                for entity_uuid in comm["member_ids"]:
                    self.graph.query(f"""
                        MATCH (e:Entity {{uuid: '{entity_uuid}'}})
                        MATCH (c:Community {{community_id: '{cid}'}})
                        MERGE (e)-[:BELONGS_TO]->(c)
                    """)

                saved += 1
                logger.debug(f"Saved community {cid}: {name} ({comm['member_count']} members)")

            except Exception as e:
                logger.warning(f"Failed to save community {cid}: {e}")

        logger.info(f"Saved {saved}/{len(communities)} communities to FalkorDB")
        return saved

    def clear_existing_communities(self) -> int:
        """
        Remove existing Community nodes and BELONGS_TO relationships.

        Useful before re-running community detection.

        Returns:
            Number of communities deleted.
        """
        try:
            # Delete relationships first
            self.graph.query("MATCH ()-[r:BELONGS_TO]->() DELETE r")

            # Delete community nodes
            result = self.graph.query("MATCH (c:Community) DELETE c RETURN count(*) as count")
            count = result.result_set[0][0] if result.result_set else 0

            logger.info(f"Deleted {count} existing communities")
            return count

        except Exception as e:
            logger.error(f"Failed to clear communities: {e}")
            return 0

    def run(
        self,
        clear_existing: bool = True,
        generate_summaries: bool = False,
    ) -> int:
        """
        Run full community detection pipeline.

        Args:
            clear_existing: Whether to remove existing communities first.
            generate_summaries: Whether to generate LLM summaries.

        Returns:
            Number of communities created.
        """
        if clear_existing:
            self.clear_existing_communities()

        communities = self.detect_communities()

        if not communities:
            logger.info("No communities detected")
            return 0

        return self.save_communities(communities, generate_summaries)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get community detection statistics.

        Returns:
            Dictionary with community statistics.
        """
        stats = {}

        try:
            # Count communities
            result = self.graph.query("MATCH (c:Community) RETURN count(c) as count")
            stats["total_communities"] = result.result_set[0][0] if result.result_set else 0

            # Community sizes
            result = self.graph.query("""
                MATCH (c:Community)
                RETURN c.community_id, c.name, c.member_count, c.modularity_score
                ORDER BY c.member_count DESC
            """)
            stats["communities"] = []
            for row in result.result_set:
                stats["communities"].append({
                    "community_id": row[0],
                    "name": row[1],
                    "member_count": row[2],
                    "modularity": row[3],
                })

            # Entities in communities vs not
            result = self.graph.query("""
                MATCH (e:Entity)
                WHERE e.fact_status = 'active'
                OPTIONAL MATCH (e)-[:BELONGS_TO]->(c:Community)
                RETURN count(DISTINCT e) as total_entities,
                       count(DISTINCT c) as entities_in_communities
            """)
            if result.result_set:
                stats["total_active_entities"] = result.result_set[0][0]
                stats["entities_in_communities"] = result.result_set[0][1]

            # Average community size
            if stats.get("communities"):
                sizes = [c["member_count"] for c in stats["communities"]]
                stats["avg_community_size"] = sum(sizes) / len(sizes)
                stats["max_community_size"] = max(sizes)
                stats["min_community_size"] = min(sizes)

        except Exception as e:
            logger.error(f"Failed to get community stats: {e}")
            stats["error"] = str(e)

        return stats


def main():
    """CLI entry point for community detection."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Detect entity communities using Louvain algorithm")

    parser.add_argument(
        "--graph",
        type=str,
        default="wikidata",
        help="FalkorDB graph name (default: wikidata)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=3,
        help="Minimum community size (default: 3)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Louvain resolution parameter (default: 1.0)",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Don't clear existing communities",
    )
    parser.add_argument(
        "--generate-summaries",
        action="store_true",
        help="Generate LLM summaries for communities",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show current statistics",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Initialize FalkorDB connection
    from falkordb import FalkorDB

    FALKORDB_HOST = os.getenv("FALKORDB_HOST", "host.docker.internal")
    FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))

    client = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    graph = client.select_graph(args.graph)

    # Initialize LLM if generating summaries
    llm = None
    if args.generate_summaries:
        try:
            from langchain_ollama import ChatOllama
            OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
            llm = ChatOllama(
                model="qwen3:8b",
                base_url=OLLAMA_HOST,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")

    # Initialize detector
    detector = EntityCommunityDetector(
        graph=graph,
        min_community_size=args.min_size,
        resolution=args.resolution,
        llm=llm,
    )

    # Stats-only mode
    if args.stats_only:
        stats = detector.get_stats()
        print(f"\nCommunity Statistics ({args.graph}):")
        print("-" * 40)
        print(f"Total communities: {stats.get('total_communities', 0)}")
        print(f"Total active entities: {stats.get('total_active_entities', 0)}")
        print(f"Entities in communities: {stats.get('entities_in_communities', 0)}")
        if stats.get("communities"):
            print(f"Avg community size: {stats.get('avg_community_size', 0):.1f}")
            print(f"Max community size: {stats.get('max_community_size', 0)}")
            print(f"Min community size: {stats.get('min_community_size', 0)}")
            print("\nTop 10 communities:")
            for c in stats["communities"][:10]:
                print(f"  {c['name']}: {c['member_count']} members")
        return

    # Run community detection
    print("\n" + "=" * 60)
    print("ENTITY COMMUNITY DETECTION")
    print("=" * 60)
    print(f"Graph:           {args.graph}")
    print(f"Min Size:        {args.min_size}")
    print(f"Resolution:      {args.resolution}")
    print(f"Clear Existing:  {not args.keep_existing}")
    print(f"Generate Summaries: {args.generate_summaries}")
    print("=" * 60)

    count = detector.run(
        clear_existing=not args.keep_existing,
        generate_summaries=args.generate_summaries,
    )

    print(f"\nCreated {count} entity communities")

    # Show stats
    stats = detector.get_stats()
    if stats.get("communities"):
        print("\nTop communities:")
        for c in stats["communities"][:5]:
            print(f"  {c['name']}: {c['member_count']} members")


if __name__ == "__main__":
    main()
