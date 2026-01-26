# GE Vernova Knowledge Graph - Test Questions

## Graph Statistics Summary

| Metric | Count |
|--------|-------|
| WikiPage nodes | 2,252 |
| Entity nodes | 409 |
| DocumentChunk nodes | 308 |
| **Total Nodes** | **2,969** |
| Total Relationships | 3,363 |

### Relationship Types
- INSTANCE_OF: 1,419
- SUBCLASS_OF: 860
- EXTRACTED_FROM: 568
- HAS_CHUNK: 308
- PART_OF: 65
- USES: 23
- IS_A, CONTAINS, INFLUENCES, MANUFACTURES, etc.

### Entity Ontology Distribution
- PhysicalObject: 172
- Concept: 50
- Quantity: 39
- Process: 32
- Resource: 28
- Location: 27
- Organization: 25

---

## Test Questions

### Category 1: Factual Questions (from Wikipedia content)

1. **What is a gas turbine and how does it work?**
   - Expected source: WikiPage "gas turbine", DocumentChunks
   - Answer should include: combustion, compressor, turbine stages, Brayton cycle

2. **What is the typical thermal efficiency of a combined-cycle power plant?**
   - Expected source: DocumentChunks from "combined cycle power station"
   - Answer should mention: ~60% efficiency, steam recovery

3. **How does a pumped-storage power station work?**
   - Expected source: WikiPage "pumped-storage power station"
   - Answer should include: upper/lower reservoirs, pumping during low demand

4. **What are the main components of a wind turbine?**
   - Expected source: WikiPage "wind turbine", related entities
   - Answer should include: rotor, nacelle, tower, generator

5. **What is the purpose of HVDC transmission?**
   - Expected source: WikiPage "HVDC power line", "HVDC converter"
   - Answer should mention: long-distance transmission, lower losses

---

### Category 2: Relationship Questions (from graph structure)

6. **What types of turbines are subclasses of the general turbine concept?**
   - Query: MATCH (t)-[:SUBCLASS_OF]->(:WikiPage {name: 'turbine'}) RETURN t.name
   - Expected: gas turbine, steam turbine, wind turbine, etc.

7. **What power generation technologies use gas turbines?**
   - Query pattern: gas turbine -[INSTANCE_OF|PART_OF]-> power station types
   - Expected: combined cycle, simple cycle, cogeneration

8. **Which entities are related to nuclear power plants?**
   - Query: MATCH (n:WikiPage {name: 'Nuclear power plant'})-[]-(related) RETURN related
   - Expected: nuclear reactor, cooling systems, steam turbine

9. **What is the relationship between HVDC and transmission grids?**
   - Expected: HVDC is used for long-distance power transmission in grids

10. **Which power plants use steam turbines?**
    - Query pattern: steam turbine relationships
    - Expected: nuclear, combined cycle, coal-fired plants

---

### Category 3: Comparative Questions

11. **How do onshore and offshore wind farms differ?**
    - Sources: WikiPages for both, extracted entities
    - Answer should include: foundation types, wind resources, maintenance access

12. **Compare gas turbines vs steam turbines for power generation.**
    - Sources: WikiPages and entities for both
    - Key differences: fuel type, efficiency, startup time, flexibility

13. **What are the advantages of HVDC over AC transmission?**
    - Sources: HVDC WikiPage, transmission grid entities
    - Answer: lower losses over distance, undersea cables, asynchronous connection

14. **How does a hydroelectric power station compare to pumped-storage?**
    - Sources: Both WikiPages
    - Key differences: storage capability, bidirectional flow, grid balancing

15. **What are the efficiency differences between simple-cycle and combined-cycle gas turbines?**
    - Sources: Combined cycle entities, gas turbine WikiPage
    - Answer: ~35% simple cycle vs ~60% combined cycle

---

### Category 4: GE Vernova Product-Relevant Questions

16. **What products and technologies are used in gas power generation?**
    - Expected coverage: gas turbines, combined cycle systems, combustion technology
    - Graph path: gas power seed entities -> related concepts

17. **How does grid electrification involve transformers and HVDC?**
    - Seeds: Transformer, HVDC, Electrical grid
    - Answer should cover: voltage transformation, long-distance transmission

18. **What nuclear technologies are represented in the knowledge graph?**
    - Seeds: Nuclear power plant, Nuclear reactor
    - Expected: reactor types, safety systems, cooling technologies

19. **What wind energy technologies are covered?**
    - Seeds: Wind turbine, Wind farm, Offshore wind farm
    - Coverage: turbine components, farm configurations, offshore challenges

20. **What hydroelectric power technologies are in the graph?**
    - Seeds: Hydroelectric power station, Pumped-storage power station
    - Expected: turbine types (Francis, Kaplan), dam systems, storage

---

## Verification Cypher Queries

```cypher
-- Count nodes by type
MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC

-- Sample WikiPage -> Chunk -> Entity path
MATCH (w:WikiPage)-[:HAS_CHUNK]->(c:DocumentChunk)<-[:EXTRACTED_FROM]-(e:Entity)
RETURN w.name, count(DISTINCT c) as chunks, count(DISTINCT e) as entities
ORDER BY entities DESC LIMIT 10

-- Find entities related to gas turbines
MATCH (w:WikiPage {name: 'gas turbine'})-[:HAS_CHUNK]->(c:DocumentChunk)
MATCH (e:Entity)-[:EXTRACTED_FROM]->(c)
RETURN DISTINCT e.name, e.ontology_type

-- Find subclasses of turbine
MATCH (sub:WikiPage)-[:SUBCLASS_OF]->(parent:WikiPage)
WHERE parent.name CONTAINS 'turbine'
RETURN sub.name, parent.name LIMIT 20

-- Find entity relationships by type
MATCH (e1:Entity)-[r]->(e2:Entity)
RETURN type(r) as rel_type, count(*) as count
ORDER BY count DESC
```

---

## Running Test Questions with agent_qa.py

```bash
# Test a factual question
python agent_qa.py --backend falkordb --graph ge_vernova --no-web \
    -q "What is a gas turbine and how does it work?"

# Test a relationship question
python agent_qa.py --backend falkordb --graph ge_vernova --no-web \
    -q "What types of turbines are used in power generation?"

# Test a comparative question
python agent_qa.py --backend falkordb --graph ge_vernova --no-web \
    -q "Compare gas turbines vs steam turbines for power generation"

# Interactive mode
python agent_qa.py --backend falkordb --graph ge_vernova -i
```

---

## Build Commands Reference

```bash
# Stage 1: Build Wikidata backbone
python wikidata_kg_builder.py \
    --seeds-file ge_vernova_seeds_small.json \
    --max-depth 3 \
    --max-per-level 100 \
    --strategy root_first \
    --backend falkordb \
    --falkordb-graph ge_vernova

# Stage 2: Process Wikipedia articles
python -m knowledge_graph.wikipedia_pipeline \
    --graph ge_vernova \
    --max-articles 50 \
    --chunk-size 1500

# Stage 3: Extract entities
python -m knowledge_graph.entity_extraction \
    --graph ge_vernova \
    --max-chunks 50 \
    --ontology medium
```
