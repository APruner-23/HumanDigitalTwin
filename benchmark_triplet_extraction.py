"""
Benchmark per valutazione Triple Extraction su dataset TinyButMighty.

Permette di:
- Valutare extraction su frasi semplici o complesse
- Variare il numero di frasi da testare (10-50)
- Cambiare modello LLM
- Calcolare metriche di similarità (Cosine, Jaccard, Fuzzy)
- Generare report dettagliato con F-measure pesata
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
from tqdm import tqdm
import argparse

# Similarity metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer

# Rich for logging
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box

# Graph
from src.agents.triplet_extraction_graph import TripletExtractionGraph
from src.config import ConfigManager


class TripletSimilarityMetrics:
    """
    Calcola similarità tra triplette estratte e ground-truth usando:
    - Cosine similarity (embeddings)
    - Jaccard similarity (token overlap)
    - Fuzzy similarity (string matching)
    """

    def __init__(self, verbose: bool = False):
        """Inizializza il modello di embeddings per Cosine similarity."""
        self.verbose = verbose
        self.console = Console()

        if verbose:
            self.console.print("[cyan]Loading SentenceTransformer model for embeddings...[/cyan]")
        else:
            print("Loading SentenceTransformer model for embeddings...")

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def triplet_to_string(self, triplet: Dict[str, Any]) -> str:
        """
        Converte una tripletta in stringa per confronto.

        Args:
            triplet: Dict con subject/predicate/object (può avere value/type)

        Returns:
            Stringa formato "subject predicate object"
        """
        # Gestisci formato con value/type (output del tuo grafo)
        if isinstance(triplet.get('subject'), dict):
            subj = triplet['subject'].get('value', '')
            pred = triplet['predicate'].get('value', '')
            obj = triplet['object'].get('value', '')
        else:
            # Formato semplice (da CSV)
            subj = str(triplet.get('subject', ''))
            pred = str(triplet.get('predicate', ''))
            obj = str(triplet.get('object', ''))

        return f"{subj} {pred} {obj}".strip().lower()

    def cosine_similarity_score(self, triplet1: Dict, triplet2: Dict) -> float:
        """
        Calcola Cosine similarity tra due triplette usando embeddings.

        Args:
            triplet1: Prima tripletta
            triplet2: Seconda tripletta

        Returns:
            Score di similarità [0, 1]
        """
        str1 = self.triplet_to_string(triplet1)
        str2 = self.triplet_to_string(triplet2)

        if not str1 or not str2:
            return 0.0

        # Genera embeddings
        emb1 = self.embedding_model.encode([str1])
        emb2 = self.embedding_model.encode([str2])

        # Cosine similarity
        sim = cosine_similarity(emb1, emb2)[0][0]
        return float(sim)

    def jaccard_similarity_score(self, triplet1: Dict, triplet2: Dict) -> float:
        """
        Calcola Jaccard similarity tra due triplette (token overlap).

        Args:
            triplet1: Prima tripletta
            triplet2: Seconda tripletta

        Returns:
            Score di similarità [0, 1]
        """
        str1 = self.triplet_to_string(triplet1)
        str2 = self.triplet_to_string(triplet2)

        if not str1 or not str2:
            return 0.0

        # Tokenizza
        tokens1 = set(str1.split())
        tokens2 = set(str2.split())

        # Jaccard = |A ∩ B| / |A ∪ B|
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def fuzzy_similarity_score(self, triplet1: Dict, triplet2: Dict) -> float:
        """
        Calcola Fuzzy similarity tra due triplette (string matching).

        Args:
            triplet1: Prima tripletta
            triplet2: Seconda tripletta

        Returns:
            Score di similarità [0, 1]
        """
        str1 = self.triplet_to_string(triplet1)
        str2 = self.triplet_to_string(triplet2)

        if not str1 or not str2:
            return 0.0

        # Fuzzy ratio normalizzato [0, 1]
        return fuzz.token_sort_ratio(str1, str2) / 100.0

    def max_similarity(
        self,
        extracted_triplet: Dict,
        ground_truth_triplets: List[Dict],
        metric: str = "cosine"
    ) -> Tuple[float, int]:
        """
        Trova la max similarità tra una tripletta estratta e le ground-truth.

        Args:
            extracted_triplet: Tripletta estratta dal modello
            ground_truth_triplets: Lista di triplette ground-truth
            metric: Metrica da usare ("cosine", "jaccard", "fuzzy")

        Returns:
            Tuple (max_score, index_best_match)
        """
        if not ground_truth_triplets:
            return 0.0, -1

        metric_fn = {
            "cosine": self.cosine_similarity_score,
            "jaccard": self.jaccard_similarity_score,
            "fuzzy": self.fuzzy_similarity_score
        }.get(metric, self.cosine_similarity_score)

        scores = [metric_fn(extracted_triplet, gt) for gt in ground_truth_triplets]
        max_idx = int(np.argmax(scores))
        max_score = scores[max_idx]

        return max_score, max_idx


class TripletExtractionBenchmark:
    """
    Benchmark per valutare Triple Extraction su dataset TinyButMighty.
    """

    def __init__(
        self,
        dataset_dir: str = "TinyButMighty/Dataset",
        llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: float = 0.3,
        enable_logging: bool = False,
        verbose: bool = False
    ):
        """
        Inizializza il benchmark.

        Args:
            dataset_dir: Path alla cartella del dataset
            llm_model: Modello LLM da usare
            temperature: Temperature per l'LLM
            enable_logging: Abilita logging dettagliato del grafo
            verbose: Abilita output dettagliato con Rich (mostra risposte LLM)
        """
        self.dataset_dir = Path(dataset_dir)
        self.llm_model = llm_model
        self.temperature = temperature
        self.enable_logging = enable_logging
        self.verbose = verbose

        # Console Rich
        self.console = Console()

        # Carica config per API key
        config = ConfigManager()
        self.api_key = config.get("groq_api_key")

        # Metrics calculator
        self.metrics = TripletSimilarityMetrics(verbose=verbose)

        # Inizializza il grafo in extraction_only mode
        if verbose:
            self.console.print(f"[cyan]Initializing TripletExtractionGraph with model: {llm_model}[/cyan]")
        else:
            print(f"Initializing TripletExtractionGraph with model: {llm_model}")

        self.graph = TripletExtractionGraph(
            llm_api_key=self.api_key,
            llm_model=llm_model,
            temperature=temperature,
            enable_logging=enable_logging,
            extraction_only=True  # SOLO EXTRACTION
        )

    def load_dataset(
        self,
        dataset_type: str = "complex",
        n_samples: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carica il dataset (semplice o complesso).

        Args:
            dataset_type: "simple" o "complex"
            n_samples: Numero di frasi da caricare (max 50 per complex)

        Returns:
            Tuple (sentences_df, triplets_df)
        """
        if dataset_type == "complex":
            sentences_file = self.dataset_dir / "complex.csv"
            triplets_file = self.dataset_dir / "triples_complex.csv"
        else:
            sentences_file = self.dataset_dir / "data.csv"
            triplets_file = self.dataset_dir / "triples.csv"

        # Carica CSV
        sentences_df = pd.read_csv(sentences_file, encoding='utf-8')
        triplets_df = pd.read_csv(triplets_file, encoding='utf-8')

        # Limita a n_samples
        if dataset_type == "complex":
            # Complex ha colonna CS001, CS002, etc.
            sentences_df = sentences_df.head(n_samples)
            # Filtra triplette corrispondenti
            sentence_ids = sentences_df.iloc[:, 0].unique()
            triplets_df = triplets_df[triplets_df['Complex_Sentence_ID'].isin(sentence_ids)]
        else:
            # Simple ha ID come CS001-SIM-1, CS001-SIM-2, etc.
            # Prendi le prime n righe delle frasi complesse e tutte le loro semplificazioni
            unique_complex_ids = sentences_df.iloc[:, 0].unique()[:n_samples]
            sentences_df = sentences_df[sentences_df.iloc[:, 0].isin(unique_complex_ids)]
            triplets_df = triplets_df[triplets_df['Complex_Sentence_ID'].isin(unique_complex_ids)]

        print(f"Loaded {len(sentences_df)} sentences and {len(triplets_df)} ground-truth triplet annotations")
        return sentences_df, triplets_df

    def parse_ground_truth_triplets(self, annotation_str: str, use_semicolon: bool = True) -> List[Dict[str, str]]:
        """
        Parsing delle triplette ground-truth dal CSV.

        Args:
            annotation_str: Stringa con annotazioni dal CSV
            use_semicolon: Se True usa ';' come separatore (complex), altrimenti ',' (simple)

        Returns:
            Lista di dict con subject/predicate/object
        """
        triplets = []
        annotation_str = str(annotation_str).strip()

        # Rimuovi spazi extra e splitta per "),("
        parts = annotation_str.split("),(")

        # Scegli separatore
        separator = ";" if use_semicolon else ","

        for part in parts:
            # Pulisci parentesi
            part = part.strip().strip("()").strip()
            if not part:
                continue

            # Split per separatore
            tokens = [t.strip() for t in part.split(separator)]

            # Deve avere almeno 3 elementi (subject, predicate, object)
            if len(tokens) >= 3:
                # Se ci sono più di 3, fai merge degli ultimi come object
                subject = tokens[0].strip()
                predicate = tokens[1].strip()
                obj = (separator + " ").join(tokens[2:]).strip()

                triplets.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })

        return triplets

    def extract_triplets_from_text(self, text: str, sentence_id: str = "") -> List[Dict[str, Any]]:
        """
        Estrae triplette da un testo usando il grafo.

        Args:
            text: Testo di input
            sentence_id: ID della frase (per logging)

        Returns:
            Lista di triplette estratte
        """
        if self.verbose:
            # Mostra input text
            self.console.print()
            self.console.print(Panel(
                f"[bold cyan]Sentence ID:[/bold cyan] {sentence_id}\n\n"
                f"[bold]Input Text:[/bold]\n{text}",
                title="🔍 Extracting Triplets",
                border_style="cyan"
            ))

        result = self.graph.run(input_text=text, chunk_size=2000)
        extracted = result.get("final_triplets", [])

        if self.verbose:
            # Mostra triplette estratte
            if extracted:
                triplets_table = Table(title="📊 Extracted Triplets", box=box.ROUNDED, border_style="green")
                triplets_table.add_column("#", style="dim", width=3)
                triplets_table.add_column("Subject", style="cyan", width=25)
                triplets_table.add_column("Predicate", style="yellow", width=20)
                triplets_table.add_column("Object", style="magenta", width=25)

                for idx, trip in enumerate(extracted, 1):
                    subj = trip.get('subject', {}).get('value', 'N/A')
                    pred = trip.get('predicate', {}).get('value', 'N/A')
                    obj = trip.get('object', {}).get('value', 'N/A')
                    triplets_table.add_row(str(idx), subj, pred, obj)

                self.console.print(triplets_table)
            else:
                self.console.print("[yellow]⚠️  No triplets extracted![/yellow]")

            self.console.print()

        return extracted

    def evaluate_sentence(
        self,
        sentence: str,
        ground_truth: List[Dict],
        sentence_id: str = "",
        metric: str = "cosine",
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Valuta extraction per una singola frase.

        Args:
            sentence: Frase di input
            ground_truth: Lista di triplette ground-truth
            sentence_id: ID della frase (per logging)
            metric: Metrica di similarità da usare
            threshold: Soglia di similarità per considerare un match valido

        Returns:
            Dict con metriche (precision, recall, f1, etc.)
        """
        # Estrai triplette
        extracted = self.extract_triplets_from_text(sentence, sentence_id)

        if not extracted:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "extracted_count": 0,
                "ground_truth_count": len(ground_truth),
                "matched_count": 0,
                "avg_similarity": 0.0
            }

        # Calcola similarità per ogni tripletta estratta
        # IMPORTANTE: Usiamo matching one-to-one (ogni GT può matchare al max 1 estratta)
        similarities = []
        matched_gt_indices = set()  # Tiene traccia delle GT già matchate
        matched_extracted = 0  # True Positives

        for ext_trip in extracted:
            max_sim, best_gt_idx = self.metrics.max_similarity(ext_trip, ground_truth, metric=metric)
            similarities.append(max_sim)

            # Match valido solo se:
            # 1. Similarity >= threshold
            # 2. Questa GT non è stata già matchata da un'altra estratta
            if max_sim >= threshold and best_gt_idx not in matched_gt_indices:
                matched_extracted += 1
                matched_gt_indices.add(best_gt_idx)

        # Metriche
        avg_sim = np.mean(similarities) if similarities else 0.0

        # True Positives = estratte che matchano una GT (one-to-one)
        # False Positives = estratte che NON matchano alcuna GT (o matchano GT già usate)
        # False Negatives = GT che NON sono matchate da alcuna estratta

        true_positives = matched_extracted
        false_positives = len(extracted) - matched_extracted
        false_negatives = len(ground_truth) - len(matched_gt_indices)

        precision = true_positives / len(extracted) if extracted else 0.0
        recall = true_positives / len(ground_truth) if ground_truth else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Log risultati se verbose
        if self.verbose:
            # Mostra ground truth
            gt_table = Table(title="📚 Ground Truth Triplets", box=box.ROUNDED, border_style="blue")
            gt_table.add_column("#", style="dim", width=3)
            gt_table.add_column("Subject", style="cyan", width=25)
            gt_table.add_column("Predicate", style="yellow", width=20)
            gt_table.add_column("Object", style="magenta", width=25)

            for idx, gt in enumerate(ground_truth, 1):
                subj = gt.get('subject', 'N/A')
                pred = gt.get('predicate', 'N/A')
                obj = gt.get('object', 'N/A')
                gt_table.add_row(str(idx), subj, pred, obj)

            self.console.print(gt_table)

            # Mostra metriche
            metrics_panel = Panel(
                f"[bold green]Precision:[/bold green] {precision:.3f} (TP={true_positives}, FP={false_positives})\n"
                f"[bold cyan]Recall:[/bold cyan]    {recall:.3f} (TP={true_positives}, FN={false_negatives})\n"
                f"[bold yellow]F1-Score:[/bold yellow]  {f1:.3f}\n"
                f"[bold magenta]Avg Similarity:[/bold magenta] {avg_sim:.3f}\n\n"
                f"[dim]Extracted: {len(extracted)} | Ground-Truth: {len(ground_truth)} | Matched (1-to-1): {true_positives}[/dim]",
                title="📊 Evaluation Metrics",
                border_style="green"
            )
            self.console.print(metrics_panel)
            self.console.print("[dim]" + "─" * 80 + "[/dim]\n")

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "extracted_count": len(extracted),
            "ground_truth_count": len(ground_truth),
            "matched_count": true_positives,  # Now this is True Positives (one-to-one)
            "avg_similarity": avg_sim,
            "extracted_triplets": extracted,
            "similarities": similarities
        }

    def run_benchmark(
        self,
        dataset_type: str = "complex",
        n_samples: int = 10,
        metric: str = "cosine",
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Esegue il benchmark completo.

        Args:
            dataset_type: "simple" o "complex"
            n_samples: Numero di frasi da testare (10-50)
            metric: Metrica di similarità ("cosine", "jaccard", "fuzzy")
            threshold: Soglia per match valido

        Returns:
            Dict con risultati aggregati
        """
        print(f"\n{'='*70}")
        print(f"Starting Benchmark: {dataset_type.upper()} sentences")
        print(f"Model: {self.llm_model}")
        print(f"Samples: {n_samples} | Metric: {metric} | Threshold: {threshold}")
        print(f"{'='*70}\n")

        # Carica dataset
        sentences_df, triplets_df = self.load_dataset(dataset_type, n_samples)

        results = []

        # Itera sulle frasi
        if dataset_type == "complex":
            # Complex: una riga = una frase complessa
            for idx, row in tqdm(sentences_df.iterrows(), total=len(sentences_df), desc="Processing"):
                sentence_id = row.iloc[0]  # ES: CS001
                sentence_text = row.iloc[1]  # Colonna con la frase

                # Ground-truth per questa frase
                gt_rows = triplets_df[triplets_df['Complex_Sentence_ID'] == sentence_id]
                gt_str = gt_rows.iloc[0]['Annotated_Triples'] if len(gt_rows) > 0 else ""
                ground_truth = self.parse_ground_truth_triplets(gt_str, use_semicolon=True)

                # Valuta
                eval_result = self.evaluate_sentence(
                    sentence_text,
                    ground_truth,
                    sentence_id=sentence_id,
                    metric=metric,
                    threshold=threshold
                )
                eval_result['sentence_id'] = sentence_id
                eval_result['sentence_text'] = sentence_text
                results.append(eval_result)

        else:
            # Simple: le frasi sono già split nella colonna 3
            # NOTA: Non possiamo associare perfettamente ogni Simple_Sentence_ID a una singola frase
            # perché data.csv ha tutte le frasi semplici in UNA cella (colonna 3)
            # Quindi valutiamo contro TUTTE le triplette del gruppo CS00X

            for idx, row in tqdm(sentences_df.iterrows(), total=len(sentences_df), desc="Processing"):
                sentence_id = row.iloc[0]  # ES: CS001
                simple_sentences = row.iloc[2]  # Colonna con TUTTE le frasi semplici (aggregate)

                # Ground-truth: raccogliamo TUTTE le triplette per questo gruppo complesso
                gt_rows = triplets_df[triplets_df['Complex_Sentence_ID'] == sentence_id]

                # Aggrega tutte le triplette del gruppo
                all_ground_truth = []
                for _, gt_row in gt_rows.iterrows():
                    gt_str = gt_row['Annotated_Triples']
                    ground_truth = self.parse_ground_truth_triplets(gt_str, use_semicolon=False)
                    all_ground_truth.extend(ground_truth)

                # Valuta UNA VOLTA per l'intero gruppo (simple_sentences contiene tutte le frasi)
                eval_result = self.evaluate_sentence(
                    simple_sentences,
                    all_ground_truth,
                    sentence_id=sentence_id,
                    metric=metric,
                    threshold=threshold
                )
                eval_result['sentence_id'] = sentence_id
                eval_result['sentence_text'] = simple_sentences
                results.append(eval_result)

        # Aggregazione
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_similarity = np.mean([r['avg_similarity'] for r in results])

        total_extracted = sum([r['extracted_count'] for r in results])
        total_ground_truth = sum([r['ground_truth_count'] for r in results])
        total_matched = sum([r['matched_count'] for r in results])

        report = {
            "dataset_type": dataset_type,
            "n_samples": n_samples,
            "model": self.llm_model,
            "metric": metric,
            "threshold": threshold,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_similarity": avg_similarity,
            "total_extracted": total_extracted,
            "total_ground_truth": total_ground_truth,
            "total_matched": total_matched,
            "per_sentence_results": results,
            "timestamp": datetime.now().isoformat()
        }

        return report

    def save_report(self, report: Dict[str, Any], output_dir: str = "benchmark_results"):
        """
        Salva il report in JSON e stampa summary.

        Args:
            report: Dict con risultati
            output_dir: Cartella output
        """
        # Crea cartella
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Nome file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_type = report['dataset_type']
        model_name = report['model'].replace("/", "_")
        filename = f"benchmark_{dataset_type}_{model_name}_{timestamp}.json"

        # Salva JSON
        output_file = output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"BENCHMARK RESULTS - {dataset_type.upper()} SENTENCES")
        print(f"{'='*70}")
        print(f"Model: {report['model']}")
        print(f"Samples: {report['n_samples']}")
        print(f"Metric: {report['metric']} | Threshold: {report['threshold']}")
        print(f"\n--- Aggregate Metrics ---")
        print(f"Average Precision: {report['avg_precision']:.3f}")
        print(f"Average Recall:    {report['avg_recall']:.3f}")
        print(f"Average F1-Score:  {report['avg_f1']:.3f}")
        print(f"Average Similarity: {report['avg_similarity']:.3f}")
        print(f"\n--- Counts ---")
        print(f"Total Extracted:     {report['total_extracted']}")
        print(f"Total Ground-Truth:  {report['total_ground_truth']}")
        print(f"Total Matched:       {report['total_matched']}")
        print(f"\nReport saved to: {output_file}")
        print(f"{'='*70}\n")


def main():
    """Entry point per CLI."""
    parser = argparse.ArgumentParser(description="Benchmark Triple Extraction on TinyButMighty dataset")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["simple", "complex"],
        default="complex",
        help="Dataset type: simple or complex sentences"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of sentences to test (10-50)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="LLM model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "jaccard", "fuzzy"],
        default="cosine",
        help="Similarity metric"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for valid match"
    )
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable detailed logging of graph execution"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with Rich (shows LLM responses, tables, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for reports"
    )

    args = parser.parse_args()

    # Inizializza benchmark
    benchmark = TripletExtractionBenchmark(
        llm_model=args.model,
        temperature=args.temperature,
        enable_logging=args.enable_logging,
        verbose=args.verbose
    )

    # Esegui
    report = benchmark.run_benchmark(
        dataset_type=args.dataset,
        n_samples=args.n_samples,
        metric=args.metric,
        threshold=args.threshold
    )

    # Salva
    benchmark.save_report(report, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
