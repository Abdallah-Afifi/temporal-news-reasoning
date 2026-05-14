"""
coref_resolver.py
==================
Build step 3 (optional): replace pronouns with their real names inside chunk text.

Simple summary
--------------
After chunking, some chunks contain pronouns like "he", "she", or "they".
Downstream NER (GLiNER) and the vector embedder work better when they see
the actual name ("Miles Jacobson") instead of a bare pronoun ("he").

This module runs ``fastcoref.FCoref`` on each chunk, finds coreference clusters
(groups of mentions that refer to the same person or thing), and replaces every
pronoun in the cluster with the first non-pronoun mention (the "root").

Grammar note: possessive pronouns like "his" become "Name's" automatically.

Temporal RAG System · Detailed Architecture v4

Position in pipeline
--------------------
    [Chunk Splitter (chunk_splitter.py)]  →  THIS MODULE (optional)  →  [GLiNER Extractor]

Usage (from repository root)
-----------------------------
    python -m temporal_rag.coref_resolver --input chunks.jsonl --output chunks_deref.jsonl

    # Process only the first 50 chunks (useful for testing):
    python -m temporal_rag.coref_resolver --input chunks.jsonl --output chunks_deref.jsonl --max-docs 50
"""
import argparse
import json
import os
from fastcoref import FCoref

class PronounResolver:
    """Replace pronouns in text with their real-world referents using coreference resolution.

    Uses ``fastcoref.FCoref`` to find all mention clusters (groups of words that
    refer to the same entity).  For each cluster, the first non-pronoun mention
    is treated as the "root name", and every pronoun in the cluster is replaced
    with that root name.

    Example
    -------
    Input:  "Miles Jacobson announced the game. He said it would launch in November."
    Output: "Miles Jacobson announced the game. Miles Jacobson said it would launch in November."
    """

    def __init__(self, device: str = 'cpu') -> None:
        """Load the FCoref model onto ``device`` (``"cpu"`` or ``"cuda"``).

        Parameters
        ----------
        device:
            Where to run the model.  Use ``"cpu"`` if no GPU is available.
        """
        print("Loading Coreference Resolution Model...")
        self.model = FCoref(device=device)

        # Only these exact words are replaced.  Verbs (e.g. "run") are never touched.
        self.PRONOUNS = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"}

    def resolve_document(self, text: str) -> str:
        """Return a copy of ``text`` with pronouns replaced by their referent names.

        How it works (step by step)
        ---------------------------
        1. Run ``FCoref`` on the text to get coreference clusters.
           Each cluster is a list of character-span pairs that all refer to the same entity.
        2. For each cluster, find the first span whose text is NOT in ``self.PRONOUNS``.
           That span becomes the "root name".
        3. For every other span in the cluster that IS a pronoun, queue a replacement:
           swap the pronoun with the root name.
           - If the pronoun is possessive (his / her / its / their), add ``'s`` to the root.
        4. Apply all replacements from back to front so earlier indices stay valid.

        Returns the original ``text`` unchanged if:
        - The text is empty.
        - FCoref finds no clusters.
        - Every mention in a cluster is itself a pronoun (no root name to use).
        """

        if not text.strip():
            return text

        # Get the character indices of all clusters
        preds = self.model.predict(texts=[text])
        clusters_indices = preds[0].get_clusters(as_strings=False)
        
        replacements = []
        
        for cluster in clusters_indices:
            # 1. Find the root noun (the first mention that is NOT a pronoun)
            root_mention = None
            for start, end in cluster:
                mention_text = text[start:end]
                if mention_text.lower() not in self.PRONOUNS:
                    root_mention = mention_text
                    break
            
            if not root_mention:
                continue # Skip if the whole cluster is just pronouns
            
            # 2. Queue up replacements ONLY if the target word is a pronoun
            for start, end in cluster:
                mention_text = text[start:end]
                
                if mention_text.lower() in self.PRONOUNS:
                    final_replacement = root_mention
                    
                    # Grammar fix: if the pronoun is possessive, make the noun possessive
                    if mention_text.lower() in {"his", "hers", "its", "their"} and not root_mention.endswith("'s"):
                        final_replacement += "'s"
                        
                    replacements.append((start, end, final_replacement))

        # 3. Sort replacements in reverse order (back-to-front) 
        # so character indices don't shift and break the string as we edit it
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        # 4. Apply the replacements
        resolved_text = text
        for start, end, replacement_text in replacements:
            resolved_text = resolved_text[:start] + replacement_text + resolved_text[end:]
            
        return resolved_text

def process_jsonl_file(
    input_path: str,
    output_path: str,
    resolver: PronounResolver,
    max_docs: int | None = None,
) -> None:
    """Stream a chunks JSONL file, resolve pronouns in each ``text`` field, and write output.

    Every other field in the JSON object is kept exactly as-is.
    Only the ``text`` field is rewritten by ``resolver.resolve_document``.

    Parameters
    ----------
    input_path:
        Path to the JSONL produced by ``chunk_splitter.py``.
    output_path:
        Path where the dereferenced JSONL will be written.
    resolver:
        A ``PronounResolver`` instance (model already loaded).
    max_docs:
        Stop after processing this many chunks.  ``None`` processes all of them.
    """
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}")
        return

    print(f"Processing {input_path} -> {output_path}...")

    processed_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if not line.strip():
                continue
            if max_docs is not None and processed_count >= max_docs:
                break

            data = json.loads(line)

            raw_text = data.get("text", "")

            if raw_text:
                clean_text = resolver.resolve_document(raw_text)
                data["text"] = clean_text

            json.dump(data, outfile, ensure_ascii=False)
            outfile.write("\n")

            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Resolved {processed_count} chunks...")

    print(f"Done. Resolved {processed_count} chunks → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pronoun coreference resolution on chunk JSONL")
    parser.add_argument(
        "--input", default="chunks.jsonl",
        help="Input JSONL from chunk_splitter.py (default: chunks.jsonl)"
    )
    parser.add_argument(
        "--output", default="chunks_deref.jsonl",
        help="Output JSONL with resolved pronouns (default: chunks_deref.jsonl)"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Process only the first N chunks (default: all)"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device for the model (default: cpu)"
    )
    args = parser.parse_args()

    resolver = PronounResolver(device=args.device)
    process_jsonl_file(args.input, args.output, resolver, max_docs=args.max_docs)