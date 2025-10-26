"""
Standalone script to generate embeddings for DAIGT samples.

Run this script OUTSIDE of Jupyter to avoid memory issues:
    python generate_embeddings_standalone.py

This will generate embedding files for existing CSV samples.
"""

import numpy as np
import pandas as pd
import os

def generate_embeddings_for_sample(csv_file, force=False):
    """Generate embeddings for a single sample CSV file."""
    print(f"\n{'='*70}")
    print(f"Processing: {csv_file}")
    print(f"{'='*70}")
    
    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"  ❌ File not found: {csv_file}")
        print("  Run prepare_daigt_samples.py first to create CSV files")
        return False
    
    # Load CSV
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df):,} texts")
    
    # Check if embeddings already exist
    embeddings_file = csv_file.replace('.csv', '_embeddings.npz')
    if os.path.exists(embeddings_file) and not force:
        print(f"  ✓ Embeddings already exist: {embeddings_file}")
        print(f"  Skipping (use --force to regenerate)")
        return True
    
    # Import sentence-transformers (only when needed)
    print("  Loading sentence-transformers model...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ❌ sentence-transformers not installed!")
        print("\n  Install with ONE of these:")
        print("    pip install sentence-transformers")
        print("    conda install -c conda-forge sentence-transformers")
        return False
    
    # Load model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"  Model loaded: {model.get_sentence_embedding_dimension()}D embeddings")
    
    # Generate embeddings in batches
    print(f"  Generating embeddings...")
    batch_size = 32
    embeddings = []
    
    for i in range(0, len(df), batch_size):
        batch_texts = df['text_normalized'].iloc[i:i+batch_size].tolist()
        batch_embeddings = model.encode(
            batch_texts, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        embeddings.append(batch_embeddings)
        
        if (i + batch_size) % 500 == 0 or i + batch_size >= len(df):
            processed = min(i + batch_size, len(df))
            print(f"    {processed:,}/{len(df):,} texts embedded")
    
    # Stack and save
    embeddings_array = np.vstack(embeddings)
    np.savez_compressed(embeddings_file, embeddings=embeddings_array)
    
    file_size_mb = os.path.getsize(embeddings_file) / (1024 * 1024)
    print(f"  ✓ Saved: {embeddings_file} ({file_size_mb:.1f} MB)")
    print(f"  ✓ Shape: {embeddings_array.shape}")
    
    return True

def main():
    import sys
    
    # Check for --force flag
    force = '--force' in sys.argv
    
    print("="*70)
    print("DAIGT Embeddings Generator (Standalone)")
    print("="*70)
    print()
    print("This script generates embeddings for existing CSV sample files.")
    print("It runs OUTSIDE Jupyter to avoid kernel crashes.")
    if force:
        print("⚠️  Running in FORCE mode - will regenerate existing embeddings")
    print()
    
    # Find all sample CSV files (auto-detect)
    import glob
    sample_files = sorted(glob.glob('data/daigt_sample_*.csv'))
    
    if not sample_files:
        print("❌ No sample CSV files found!")
        print("\nRun this first:")
        print("  python prepare_daigt_samples.py")
        return
    
    print(f"Found {len(sample_files)} sample file(s):")
    for f in sample_files:
        size_kb = os.path.getsize(f) / 1024
        print(f"  - {f} ({size_kb:.0f} KB)")
    print()
    
    # Process each file
    success_count = 0
    for csv_file in sample_files:
        if generate_embeddings_for_sample(csv_file, force=force):
            success_count += 1
    
    print()
    print("="*70)
    if success_count == len(sample_files):
        print("✓ All embeddings generated successfully!")
    elif success_count > 0:
        print(f"✓ {success_count}/{len(sample_files)} embeddings generated")
    else:
        print("❌ No embeddings generated")
    print("="*70)
    print()
    print("Next step: Run the Jupyter notebook")
    print("  The notebook will load these pre-computed embeddings")
    print()
    print("Usage:")
    print("  python generate_embeddings_standalone.py          # Skip existing")
    print("  python generate_embeddings_standalone.py --force  # Regenerate all")

if __name__ == '__main__':
    main()

