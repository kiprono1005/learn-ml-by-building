# Regularization & Generalization Lecture - Environment Setup (Isolated)

This guide sets up a dedicated virtual environment for the Regularization & Generalization lecture only. It won’t affect your shared `ml_lectures_env`.

## Version Policy
- **Lectures (existing)**: use Python 3.9 for maximum compatibility with older notebooks and packages.
- **New projects**: use Python 3.11 for better performance and long-term support.

This page focuses on the lecture's 3.9 environment. See the end for notes.

## Location
- Lecture folder: `Lecture 7 Regularization and Generalization/`
- Env path: `Lecture 7 Regularization and Generalization/regularization_env/`
- Jupyter kernel name: `Python (Regularization Py39)`

## Prerequisites
- macOS
- Python 3.9 (this lecture venv uses a Python 3.9 interpreter)

## Create and Activate the Environment
```bash
# From the repo root (or cd into the lecture folder)
# Recommended: create the venv using your Python 3.9 interpreter
"/Users/ming/Dropbox/learn-ml-by-building/ml_lectures_env/bin/python" -m venv "Lecture 7 Regularization and Generalization/regularization_env"

# Activate (macOS)
source "Lecture 7 Regularization and Generalization/regularization_env/bin/activate"

# Upgrade build tools
python -m pip install --upgrade pip setuptools wheel
```

## Install Core Packages
```bash
# Pin NumPy < 2 for compatibility across scientific stack
python -m pip install \
  "numpy<2" \
  "pandas>=1.5,<2.2" \
  "scikit-learn>=1.3,<1.5" \
  "matplotlib>=3.7,<3.9" \
  "seaborn>=0.12.2,<0.14" \
  "ipywidgets>=8.1,<8.2"
```

## Install TensorFlow (Required for this lecture)
```bash
# Apple Silicon (M1/M2/M3)
python -m pip install "tensorflow-macos>=2.16,<2.18" "tensorflow-metal>=1.1"
```

On Intel mac/Windows/Linux:
```bash
python -m pip install "tensorflow>=2.16,<2.18"
```

### Optional (extras used in some demos)
```bash
# Finance data helper (only if you need it)
python -m pip install yfinance
```

## Jupyter and Kernel Registration
```bash
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name regularization-py39 --display-name "Python (Regularization Py39)"
```

## Start Jupyter in the Lecture Folder
```bash
# From repo root
"Lecture 7 Regularization and Generalization/regularization_env/bin/jupyter" notebook --no-browser --ip=127.0.0.1
```
Open the printed URL (e.g., http://127.0.0.1:8890/tree) and select the kernel: `Python (Regularization Py39)`.

## RISE (Slide Show) in this Environment
RISE works with the classic Notebook interface (Notebook 6).
```bash
python -m pip install "notebook==6.5.7" "rise>=5.7"
python -m jupyter nbextension install rise --py --sys-prefix
python -m jupyter nbextension enable rise --py --sys-prefix
```

## Notes
- Your shared env `ml_lectures_env` remains untouched.
- If you encounter errors mentioning `np.complex_` or other NumPy-2.0 removals, ensure you have `numpy<2` in this environment.
