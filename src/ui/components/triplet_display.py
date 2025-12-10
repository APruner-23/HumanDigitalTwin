"""
Components for displaying triplets and validation results.
"""

import streamlit as st
from typing import Dict, List


def display_triplet(triplet: Dict, idx: int) -> None:
    """
    Display a single triplet with value and type rows.

    Args:
        triplet: Triplet dictionary with subject, predicate, object
        idx: Index number for the triplet
    """
    subject = triplet.get('subject', {})
    predicate = triplet.get('predicate', {})
    obj = triplet.get('object', {})

    s_val = subject.get('value', '') if isinstance(subject, dict) else subject
    p_val = predicate.get('value', '') if isinstance(predicate, dict) else predicate
    o_val = obj.get('value', '') if isinstance(obj, dict) else obj

    s_type = subject.get('type', 'N/A') if isinstance(subject, dict) else 'N/A'
    p_type = predicate.get('type', 'N/A') if isinstance(predicate, dict) else 'N/A'
    o_type = obj.get('type', 'N/A') if isinstance(obj, dict) else 'N/A'

    with st.expander(f"Tripletta #{idx}: {s_val} → {p_val} → {o_val}"):
        st.markdown("**Valori (Instance):**")
        st.code(f"{s_val}  →  {p_val}  →  {o_val}")

        st.markdown("**Tipi (Class):**")
        st.code(f"{s_type}  →  {p_type}  →  {o_type}")


def display_triplets_list(triplets: List[Dict], title: str, max_display: int = 10) -> None:
    """
    Display a list of triplets with expandable views.

    Args:
        triplets: List of triplet dictionaries
        title: Title for the section
        max_display: Maximum number of triplets to display
    """
    st.subheader(f"{title}: {len(triplets)}")

    if not triplets:
        st.info("Nessuna tripletta disponibile")
        return

    for idx, triplet in enumerate(triplets[:max_display], 1):
        display_triplet(triplet, idx)

    if len(triplets) > max_display:
        st.info(f"Mostrate prime {max_display} triplette. Totale: {len(triplets)}")

    with st.expander("🔍 Visualizza JSON completo"):
        st.json(triplets)


def display_validation_result(result: Dict, idx: int, show_branches: bool = True) -> None:
    """
    Display a validation result with matching info and branch strategies.

    Args:
        result: Validation result dictionary
        idx: Index number
        show_branches: Whether to show branch strategy comparison
    """
    score = result.get('mu', 0.0)

    with st.expander(f"Tripletta #{idx} - Score: {score:.3f}"):
        # Original triplet
        s_val = result['subject']['value']
        p_val = result['predicate']['value']
        o_val = result['object']['value']

        st.markdown(f"**Tripletta:** `{s_val}` → `{p_val}` → `{o_val}`")

        # Matching info (2x3 matrix: LLM types + Schema.org types)
        col_s, col_p, col_o = st.columns(3)

        with col_s:
            st.markdown("**🎯 Subject**")
            orig_type = result['subject'].get('original_type')
            if orig_type:
                st.markdown(f"LLM Type: `{orig_type}`")
            st.markdown(f"Schema.org: `{result['subject']['matched_class']}`")
            st.markdown(f"Score: `{result['subject']['confidence']:.3f}`")

        with col_p:
            st.markdown("**🔗 Predicate**")
            orig_type = result['predicate'].get('original_type')
            if orig_type:
                st.markdown(f"LLM Type: `{orig_type}`")
            st.markdown(f"Schema.org: `{result['predicate']['matched_property']}`")
            st.markdown(f"Score: `{result['predicate']['confidence']:.3f}`")

        with col_o:
            st.markdown("**📍 Object**")
            orig_type = result['object'].get('original_type')
            if orig_type:
                st.markdown(f"LLM Type: `{orig_type}`")
            st.markdown(f"Schema.org: `{result['object']['matched_class']}`")
            st.markdown(f"Score: `{result['object']['confidence']:.3f}`")

        # Branch path
        st.markdown(f"**🌳 Path:** {result.get('branch_path', 'N/A')}")

        # Branch strategies comparison
        if show_branches:
            with st.expander("🌳 Branch Strategies (3 metodi di esplorazione)"):
                all_branches = result.get('all_branches', [])
                if all_branches:
                    # Group by method_used
                    predicate_driven = [b for b in all_branches if b.get('method_used') == 'predicate_driven']
                    subject_driven = [b for b in all_branches if b.get('method_used') == 'subject_driven']
                    object_driven = [b for b in all_branches if b.get('method_used') == 'object_driven']

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**🔗 Predicate-Driven**")
                        st.caption("Predicate → Subject/Object")
                        if predicate_driven:
                            best = predicate_driven[0]
                            st.metric("Score μ", f"{best['mu']:.3f}")
                            st.caption(f"Path: {best.get('branch_path', 'N/A')}")
                        else:
                            st.caption("Nessun match trovato")

                    with col2:
                        st.markdown("**🎯 Subject-Driven**")
                        st.caption("Subject → Predicate → Object")
                        if subject_driven:
                            best = subject_driven[0]
                            st.metric("Score μ", f"{best['mu']:.3f}")
                            st.caption(f"Path: {best.get('branch_path', 'N/A')}")
                        else:
                            st.caption("Nessun match trovato")

                    with col3:
                        st.markdown("**📍 Object-Driven**")
                        st.caption("Object → Predicate → Subject")
                        if object_driven:
                            best = object_driven[0]
                            st.metric("Score μ", f"{best['mu']:.3f}")
                            st.caption(f"Path: {best.get('branch_path', 'N/A')}")
                        else:
                            st.caption("Nessun match trovato")

                    st.markdown("---")
                    st.markdown(f"**✅ Strategia scelta:** `{result.get('method_used', 'N/A')}` (score più alto)")

        # Top candidates
        with st.expander("🔍 Top Candidates"):
            st.markdown("**Subject candidates:**")
            for name, score in result['subject']['top_candidates'][:3]:
                st.markdown(f"- {name}: {score:.3f}")

            st.markdown("**Predicate candidates:**")
            for name, score in result['predicate']['top_candidates'][:3]:
                st.markdown(f"- {name}: {score:.3f}")

            st.markdown("**Object candidates:**")
            for name, score in result['object']['top_candidates'][:3]:
                st.markdown(f"- {name}: {score:.3f}")
