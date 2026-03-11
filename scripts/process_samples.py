import json
import os
import argparse
from pathlib import Path

from src.rag.heideltime_wrapper import HeidelTimeWrapper

def process_samples(input_file: str, output_file: str):
    print(f"Loading samples from {input_file}...")
    wrapper = HeidelTimeWrapper()
    processed_chunks = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for index, line in enumerate(lines):
        if not line.strip():
            continue
            
        data = json.loads(line)
        text = data.get("text", "")
        pub_date = data.get("published_date", "")
        
        # Ensure we have a valid anchor date
        if not pub_date:
            print(f"Warning: Chunk {index} is missing 'published_date'. Skipping temporal extraction.")
            data["T_start"] = None
            data["T_end"] = None
        else:
            print(f"Processing chunk {index}/{len(lines)} (Length: {len(text)} chars)...")
            t_start, t_end = wrapper.extract_temporal_bounds(text, document_creation_time=pub_date)
            data["T_start"] = t_start
            data["T_end"] = t_end
            print(f"  -> Extracted Bounds: [{t_start}, {t_end}] (DCT: {pub_date})")
            
        processed_chunks.append(data)
        
    print(f"Saving {len(processed_chunks)} processed chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write out pretty-printed JSON structure
        json.dump(processed_chunks, f, indent=4, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL chunks through HeidelTime to extract T_start and T_end bounding boxes.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()
    
    process_samples(args.input, args.output)
