"""
Preprocessing script: Create sampled DAIGT datasets for Lecture 12

This script loads the full DAIGT V3 dataset, creates stratified samples,
and saves them as CSV files for direct loading in the notebook.

Run this once to generate the sample datasets:
    python prepare_daigt_samples.py

Outputs:
    - daigt_sample_200.csv: 200 texts per prompt (~3k total)
    - daigt_sample_500.csv: 500 texts per prompt (~7.5k total)
    - daigt_sample_1000.csv: 1000 texts per prompt (~15k total)
"""

import numpy as np
import pandas as pd
import re

SEED = 42
np.random.seed(SEED)

# Text Normalization
def normalize_text(text):
    """Remove non-ASCII characters for consistent embedding generation."""
    if pd.isna(text):
        return ""
    return re.sub(r'[^\x20-\x7E]+', '', str(text))

# Model Extraction
def extract_model(row):
    """
    Extract standardized model name from dataset columns.
    Returns lowercase names: human, gpt, mistral, llama, etc.
    """
    if row['label'] == 0:
        return 'human'
    
    # Prefer explicit model column if available
    if 'model' in row and pd.notna(row['model']) and row['model'] != 'human':
        return row['model']
    
    # Otherwise parse from source string
    source = str(row['source']).lower()
    
    # Map common patterns to model families
    if 'gpt' in source or 'chat_gpt' in source:
        return 'gpt'
    elif 'mistral' in source:
        return 'mistral'
    elif 'llama' in source:
        return 'llama'
    elif 'falcon' in source:
        return 'falcon'
    elif 'claude' in source:
        return 'claude'
    elif 'palm' in source:
        return 'palm'
    elif 'davinci' in source:
        return 'davinci'
    elif 'curie' in source:
        return 'curie'
    elif 'babbage' in source:
        return 'babbage'
    elif 'ada' in source:
        return 'ada'
    elif 'cohere' in source:
        return 'cohere'
    else:
        return 'other'

def create_sample(df, texts_per_prompt, output_file, include_embeddings=False):
    """Create stratified sample and save to CSV."""
    print(f"\nCreating sample with {texts_per_prompt} texts per prompt...")
    
    df_sample = df.groupby('prompt_name', group_keys=False).apply(
        lambda x: x.sample(n=min(texts_per_prompt, len(x)), random_state=SEED)
    ).reset_index(drop=True)
    
    print(f"  Sample size: {len(df_sample):,} texts")
    print(f"  Prompts: {df_sample['prompt_name'].nunique()}")
    print(f"  Models: {df_sample['model'].nunique()}")
    print(f"  Human: {(df_sample['label']==0).sum():,}, AI: {(df_sample['label']==1).sum():,}")
    
    # Generate embeddings if requested
    if include_embeddings:
        print(f"  Generating embeddings (this may take a few minutes)...")
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print(f"    Model loaded: {model.get_sentence_embedding_dimension()}D embeddings")
            
            # Process in small batches to avoid memory issues
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(df_sample), batch_size):
                batch_texts = df_sample['text_normalized'].iloc[i:i+batch_size].tolist()
                batch_embeddings = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
                
                if (i + batch_size) % 500 == 0 or i + batch_size >= len(df_sample):
                    print(f"    Embedded {min(i + batch_size, len(df_sample)):,}/{len(df_sample):,} texts")
            
            # Stack all embeddings
            embeddings_array = np.vstack(embeddings)
            
            # Save embeddings as separate .npz file (much faster than CSV)
            embeddings_file = output_file.replace('.csv', '_embeddings.npz')
            np.savez_compressed(embeddings_file, embeddings=embeddings_array)
            print(f"  ✓ Embeddings saved to {embeddings_file}")
            
        except ImportError:
            print("  ⚠️  sentence-transformers not installed, skipping embeddings")
            print("     Install with: pip install sentence-transformers")
    
    # Save CSV (without embeddings to keep file size reasonable)
    df_sample.to_csv(output_file, index=False)
    print(f"  ✓ Text data saved to {output_file}")
    
    return df_sample

def main():
    print("="*70)
    print("DAIGT Dataset Preprocessing")
    print("="*70)
    
    dataset_path = 'data/daigt-v3-train-dataset/'
    
    # Load with memory-efficient dtypes
    print("\n1. Loading full DAIGT V3 dataset...")
    df1 = pd.read_csv(dataset_path + 'train_v3_drcat_01.csv', 
                      dtype={'label': 'int8', 'source': 'category', 'prompt_name': 'category'})
    df2 = pd.read_csv(dataset_path + 'train_v3_drcat_02.csv',
                      dtype={'label': 'int8', 'source': 'category', 'prompt_name': 'category'})
    print(f"   Loaded: {len(df1):,} + {len(df2):,} = {len(df1)+len(df2):,} texts")
    
    # Combine
    df_full = pd.concat([df1, df2], ignore_index=True)
    df_full = df_full[df_full['text'].notna()].copy()
    del df1, df2
    
    # Preprocess
    print("\n2. Preprocessing...")
    df_full['text_normalized'] = df_full['text'].apply(normalize_text)
    df_full['word_count'] = df_full['text_normalized'].str.split().str.len()
    df_full['model'] = df_full.apply(extract_model, axis=1)
    
    # Remove very short texts
    df_full = df_full[df_full['word_count'] > 50].copy()
    
    print(f"   After preprocessing: {len(df_full):,} texts")
    print(f"   Prompts: {df_full['prompt_name'].nunique()}")
    print(f"   Models: {df_full['model'].nunique()}")
    
    # Create multiple sample sizes
    print("\n3. Creating stratified samples...")
    print("   (Embeddings will be generated separately by generate_embeddings_standalone.py)")
    
    # Standard sizes
    sample_100 = create_sample(df_full, 100, 'data/daigt_sample_100.csv', include_embeddings=False)
    sample_200 = create_sample(df_full, 200, 'data/daigt_sample_200.csv', include_embeddings=False)
    sample_300 = create_sample(df_full, 300, 'data/daigt_sample_300.csv', include_embeddings=False)
    sample_500 = create_sample(df_full, 500, 'data/daigt_sample_500.csv', include_embeddings=False)
    sample_750 = create_sample(df_full, 750, 'data/daigt_sample_750.csv', include_embeddings=False)
    sample_1000 = create_sample(df_full, 1000, 'data/daigt_sample_1000.csv', include_embeddings=False)
    sample_2000 = create_sample(df_full, 2000, 'data/daigt_sample_2000.csv', include_embeddings=False)
    
    print("\n" + "="*70)
    print("✓ Preprocessing complete!")
    print("="*70)
    print("\nGenerated sample files:")
    print("  - daigt_sample_100.csv   (~1.5k texts, ultra-lightweight)")
    print("  - daigt_sample_200.csv   (~3k texts, safe for all systems)")
    print("  - daigt_sample_300.csv   (~4.5k texts, good for testing)")
    print("  - daigt_sample_500.csv   (~7.5k texts, recommended balance)")
    print("  - daigt_sample_750.csv   (~11k texts, more data)")
    print("  - daigt_sample_1000.csv  (~15k texts, comprehensive)")
    print("  - daigt_sample_2000.csv  (~30k texts, for powerful machines)")
    print("\nNext step: Generate embeddings")
    print("  python generate_embeddings_standalone.py")
    print("\nThen use in notebook:")
    print("  SAMPLE_FILE = 'daigt_sample_500.csv'")
    print("="*70)

if __name__ == '__main__':
    main()

