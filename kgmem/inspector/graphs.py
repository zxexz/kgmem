"""Small graph used by the demo and inspector."""

from kgmem.graph import KnowledgeGraph


def demo_graph() -> KnowledgeGraph:
    """Small example graph shared with the inspector."""
    kg = KnowledgeGraph(
        entities=["Velmora", "Quistan", "Drakenhold", "Pellin", "Sorvik", "Zennar", "Orbek", "Talia", "Brimwen"],
        relations=["capital", "currency", "located in", "lives in", "leader", "friend"])
    for h, r, t in [("Velmora", "capital", "Drakenhold"), ("Quistan", "capital", "Sorvik"),
                    ("Drakenhold", "located in", "Velmora"), ("Pellin", "located in", "Velmora"),
                    ("Sorvik", "located in", "Quistan"), ("Velmora", "currency", "Zennar"),
                    ("Quistan", "currency", "Orbek"), ("Talia", "lives in", "Pellin"),
                    ("Brimwen", "lives in", "Sorvik"), ("Velmora", "leader", "Brimwen"),
                    ("Talia", "friend", "Brimwen")]:
        kg.add(h, r, t)
    return kg
