# Setup & Run

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Initialize Git + DVC
```bash
git init
dvc init
```

## 3. Run the pipeline
```bash
dvc repro
```

## 4. View the DAG
```bash
dvc dag
```

## 5. View metrics
```bash
dvc metrics show
```

## 6. Commit to Git
```bash
git add .
git commit -m "Initial DVC pipeline"
```

---

**Re-run after changing code/params:**
```bash
dvc repro
```

**Force re-run everything:**
```bash
dvc repro -f
```
