# test_goesTo_vs_properties.py

import os
import numpy as np

from sentence_transformers import SentenceTransformer
from src.ontology.ontology_loader import SchemaOrgLoader


SCHEMA_PATH = "data/ontology/schema.jsonld"  # adatta se diverso
QUERY_PREDICATE = "recommends"


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    dot = float(np.dot(v1, v2))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def build_property_text_system(property_name: str, ontology: SchemaOrgLoader) -> str:
    """
    Costruisce il testo arricchito della proprietà, nello spirito di _get_property_embedding:
      - nome
      - descrizione (label + comment)
      - dominio (Used with: ...)
      - range (Points to: ...)
    """
    prop_desc = ontology.get_property_description(property_name)
    prop_info = ontology.get_property_info(property_name)

    parts = [property_name]

    if prop_desc:
        parts.append(prop_desc)

    domain_classes = prop_info.get("domainIncludes", [])
    if domain_classes:
        domain_str = ", ".join(domain_classes[:3])
        parts.append(f"Used with: {domain_str}")

    range_classes = prop_info.get("rangeIncludes", [])
    if range_classes:
        range_str = ", ".join(range_classes[:3])
        parts.append(f"Points to: {range_str}")

    return ". ".join(parts)


def main():
    print("=== Carico ontologia ===")
    ontology = SchemaOrgLoader(schema_path=SCHEMA_PATH)
    all_properties = ontology.get_all_properties()
    print(f"Proprietà totali in ontologia: {len(all_properties)}")

    print("\n=== Carico modello MiniLM ===")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    print(f"Modello caricato: {model_name}")

    # -----------------------------
    # TEST 1: algoritmo semplice (solo nome proprietà)
    # -----------------------------
    print("\n================ TEST 1: semplice (nome proprietà) ================")
    print(f"Query predicate: '{QUERY_PREDICATE}'\n")

    query_emb_simple = model.encode([QUERY_PREDICATE])[0]

    # Embedding di tutti i NOME delle proprietà (es: 'location', 'follows', ecc.)
    prop_name_list = all_properties
    prop_name_embs = model.encode(prop_name_list, batch_size=64)

    sims_simple = []
    for name, emb in zip(prop_name_list, prop_name_embs):
        sim = cosine_similarity(query_emb_simple, emb)
        sims_simple.append((name, sim))

    sims_simple.sort(key=lambda x: x[1], reverse=True)

    print("Top 20 proprietà (TEST 1 - semplice, solo nome):")
    for name, score in sims_simple[:20]:
        print(f"{name:<30} {score:.4f}")

    # Salviamo anche lo score specifico di 'follows' se esiste
    follows_score_simple = next((s for n, s in sims_simple if n == "follows"), None)
    if follows_score_simple is not None:
        print(f"\n[TEST 1] Similarità 'goesTo' vs 'follows' (solo nome): {follows_score_simple:.4f}")
    else:
        print("\n[TEST 1] Proprietà 'follows' non trovata tra i nomi delle proprietà.")

    # -----------------------------
    # TEST 2: algoritmo stile sistema (testo arricchito proprietà)
    # -----------------------------
    print("\n================ TEST 2: stile sistema (testo arricchito) ================")
    print(f"Query predicate: '{QUERY_PREDICATE}'\n")

    # Nel sistema, per il predicato di solito si usa il testo così com'è (o con context opzionale).
    # Qui usiamo "goesTo" puro, senza context, per vedere che succede.
    query_emb_system = model.encode([QUERY_PREDICATE])[0]

    # Costruiamo il testo arricchito di ogni proprietà come in _get_property_embedding
    enriched_texts = []
    for prop_name in all_properties:
        text = build_property_text_system(prop_name, ontology)
        enriched_texts.append(text)

    prop_enriched_embs = model.encode(enriched_texts, batch_size=16)

    sims_system = []
    for prop_name, emb in zip(all_properties, prop_enriched_embs):
        sim = cosine_similarity(query_emb_system, emb)
        sims_system.append((prop_name, sim))

    sims_system.sort(key=lambda x: x[1], reverse=True)

    print("Top 20 proprietà (TEST 2 - testo arricchito):")
    for name, score in sims_system[:20]:
        print(f"{name:<30} {score:.4f}")

    follows_score_system = next((s for n, s in sims_system if n == "follows"), None)
    if follows_score_system is not None:
        print(f"\n[TEST 2] Similarità 'goesTo' vs 'follows' (testo arricchito): {follows_score_system:.4f}")
    else:
        print("\n[TEST 2] Proprietà 'follows' non trovata tra le proprietà.")

    print("\n=== Fine test ===")


if __name__ == "__main__":
    main()