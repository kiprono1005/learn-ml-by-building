# Data Directory for Lecture 12: PCA

This directory contains all datasets and preprocessing scripts for the PCA lecture notebook.

## Directory Structure

```
data/
├── README.md                           # This file
├── daigt-v3-train-dataset/            # Original DAIGT dataset (130k texts)
│   ├── train_v3_drcat_01.csv         # Part 1 of full dataset
│   └── train_v3_drcat_02.csv         # Part 2 of full dataset
├── daigt_sample_*.csv                 # Pre-generated sample datasets
├── daigt_sample_*_embeddings.npz      # Pre-computed embeddings for samples
├── mammoth_data.csv                   # 3D mammoth skeleton dataset
├── prepare_daigt_samples.py           # Script to generate sample datasets
└── generate_embeddings_standalone.py  # Script to generate embeddings
```

## Datasets

### 1. DAIGT V3 Dataset
- **Source**: [Kaggle - DAIGT V3 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v3-train-dataset)
- **Description**: Detecting AI Generated Text dataset with 130k texts from multiple AI models and human writers
- **Location**: `daigt-v3-train-dataset/`
- **Size**: ~300 MB (2 CSV files)

### 2. Sample Datasets
Pre-generated stratified samples for different system capabilities:
- `daigt_sample_100.csv` - ~1.5k texts (ultra-lightweight)
- `daigt_sample_200.csv` - ~3k texts (safe for all systems)
- `daigt_sample_300.csv` - ~4.5k texts (good for testing)
- `daigt_sample_500.csv` - ~7.5k texts (recommended balance)
- `daigt_sample_750.csv` - ~11k texts (more data)
- `daigt_sample_1000.csv` - ~15k texts (comprehensive)
- `daigt_sample_2000.csv` - ~30k texts (for powerful machines)

Each sample has a corresponding `*_embeddings.npz` file with pre-computed 384D embeddings.

### 3. Mammoth Dataset
- **Source**: [PaCMAP Repository](https://github.com/YingfanWang/PaCMAP/tree/master/demo)
- **Description**: 10k-point 3D digitization of a mammoth skeleton
- **Purpose**: Tests global structure preservation in dimensionality reduction
- **Location**: `mammoth_data.csv`
- **Size**: ~300 KB

## Scripts

### prepare_daigt_samples.py
Generates stratified sample datasets from the full DAIGT dataset.

**Usage:**
```bash
# Run from the Lecture 12 PCA directory
python data/prepare_daigt_samples.py
```

**What it does:**
1. Loads the full DAIGT dataset (130k texts)
2. Normalizes text (removes non-ASCII characters)
3. Extracts model information
4. Creates stratified samples by prompt
5. Saves sample CSV files to `data/`

### generate_embeddings_standalone.py
Generates sentence embeddings for sample datasets.

**Usage:**
```bash
# Run from the Lecture 12 PCA directory
python data/generate_embeddings_standalone.py          # Skip existing
python data/generate_embeddings_standalone.py --force  # Regenerate all
```

**What it does:**
1. Finds all `daigt_sample_*.csv` files
2. Loads sentence-transformers model (all-MiniLM-L6-v2)
3. Generates 384D embeddings in batches
4. Saves compressed embeddings as `.npz` files

**Note:** Run this OUTSIDE Jupyter to avoid memory issues.

## Usage in Notebook

The notebook automatically loads data from the `data/` directory:

```python
# Load sample dataset
SAMPLE_FILE = 'data/daigt_sample_1000.csv'
df = pd.read_csv(SAMPLE_FILE)

# Load embeddings
embeddings_file = SAMPLE_FILE.replace('.csv', '_embeddings.npz')
embeddings_data = np.load(embeddings_file)
X = embeddings_data['embeddings']

# Load mammoth dataset
mammoth_df = pd.read_csv('data/mammoth_data.csv')
```

## First-Time Setup

If you're setting up this notebook for the first time:

1. **Download DAIGT dataset** from Kaggle and place in `data/daigt-v3-train-dataset/`
2. **Generate samples**: `python data/prepare_daigt_samples.py`
3. **Generate embeddings**: `python data/generate_embeddings_standalone.py`
4. **Run the notebook**: All data will be loaded from `data/`

## File Sizes

Approximate sizes:
- Full DAIGT dataset: ~300 MB
- Sample CSVs: 7-140 MB each
- Embedding files: 2-43 MB each
- Mammoth dataset: 300 KB
- **Total**: ~1-2 GB (depending on which samples you generate)

## Notes

- The full DAIGT dataset is NOT included in the repository (too large)
- Sample datasets and embeddings are pre-generated for convenience
- All file paths in the notebook reference the `data/` directory
- Scripts are designed to run from the parent directory (Lecture 12 PCA)
