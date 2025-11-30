# 🚀 Kaggle Training Guide - Hallucination Detection Model

Complete step-by-step guide for training your hallucination detection model on Kaggle.

---

## 📋 Prerequisites

- ✅ Kaggle account (free) - [Sign up here](https://www.kaggle.com/account/login)
- ✅ Phone verification (required for GPU access)
- ✅ Prepared datasets (ambiguity datasets downloaded)

---

## 🎯 Two Approaches

### Approach A: Upload Pre-Prepared Dataset (RECOMMENDED - Faster)
**Time**: ~20 min setup + 7-8 hours training

1. Prepare data locally
2. Upload to Kaggle Dataset
3. Run training notebook

### Approach B: Prepare Data on Kaggle
**Time**: ~10 min setup + 10 min data prep + 7-8 hours training

1. Upload raw datasets
2. Run data preparation on Kaggle
3. Run training

**Recommendation**: **Approach A** (faster, cleaner)

---

## 📦 APPROACH A: Upload Pre-Prepared Dataset

### Step 1: Prepare Data Locally (Once)

On your local machine:

```bash
cd /home/talha/projects/rag-project

# Install dependencies (if not done)
poetry install

# Prepare training data
python scripts/prepare_training_data.py \
    --config config/base_config.yaml \
    --output-dir data/training/nli_dataset \
    --balance-classes
```

**Output**: `data/training/nli_dataset/` with 3 files:
- `train.jsonl` (~71K examples, ~100 MB)
- `val.jsonl` (~8.4K examples, ~12 MB)
- `test.jsonl` (~4.2K examples, ~6 MB)

**Total size**: ~120 MB

### Step 2: Create Kaggle Dataset

1. **Zip the dataset**:
   ```bash
   cd data/training
   zip -r nli_dataset.zip nli_dataset/
   ```

2. **Go to Kaggle**:
   - Navigate to: https://www.kaggle.com/datasets
   - Click **"New Dataset"**

3. **Upload**:
   - Title: `hallucination-nli-dataset`
   - Subtitle: `NLI dataset for hallucination detection (84K examples)`
   - Upload: `nli_dataset.zip`
   - Visibility: **Private** (or Public if you want)
   - Click **"Create"**

4. **Note the dataset path**:
   - Will be: `/kaggle/input/hallucination-nli-dataset/`

### Step 3: Create Kaggle Notebook

1. **Create new notebook**:
   - Go to: https://www.kaggle.com/code
   - Click **"New Notebook"**

2. **Settings** (top-right gear icon):
   - ✅ **Accelerator**: GPU T4 x2 (or P100)
   - ✅ **Internet**: On
   - ✅ **Environment**: Python 3.10+

3. **Add dataset**:
   - Click **"+ Add Data"** (right panel)
   - Search: `hallucination-nli-dataset` (your dataset)
   - Click **"Add"**

4. **Upload notebook code**:
   - Option 1: Copy-paste from `notebooks/train_hallucination_kaggle.ipynb`
   - Option 2: File → Import Notebook → Upload `.ipynb` file

### Step 4: Update Dataset Path

In the notebook, find this cell:

```python
kaggle_dataset_path = Path('/kaggle/input/hallucination-nli-dataset')  # Change to your dataset name
```

Update to match your dataset name (usually auto-filled).

### Step 5: Run Training

1. **Click "Run All"** or Ctrl+Enter through cells
2. **Verify GPU**:
   ```
   Tesla T4, 15GB (usually)
   ```
3. **Wait** ~7-8 hours

**What happens**:
- ✅ Installs dependencies (~2 min)
- ✅ Clones your repo
- ✅ Loads dataset from `/kaggle/input/`
- ✅ Trains for 5 epochs (~7-8 hours)
- ✅ Saves checkpoints to `/kaggle/working/`
- ✅ Evaluates model
- ✅ Exports for production

### Step 6: Download Model

After training completes:

1. **Go to Output tab** (right panel)
2. **Download**:
   - `hallucination_detector.tar.gz` (~1.5 GB)
   - `evaluation_results/` (metrics, plots)

3. **Extract locally**:
   ```bash
   cd /home/talha/projects/rag-project
   tar -xzf ~/Downloads/hallucination_detector.tar.gz
   mv hallucination_detector models/
   ```

4. **Verify**:
   ```bash
   ls -lh models/hallucination_detector/
   # Should see: model/, tokenizer/, config.json
   ```

---

## 📦 APPROACH B: Prepare Data on Kaggle

### Step 1: Upload Raw Datasets to Kaggle

1. **Zip ambiguity datasets**:
   ```bash
   cd /home/talha/projects/rag-project
   zip -r ambiguity_datasets.zip data/ambiguity_datasets/
   ```

2. **Create Kaggle Dataset**:
   - Title: `ambiguity-datasets`
   - Upload: `ambiguity_datasets.zip`

3. **Create notebook and add dataset**

### Step 2: Modify Notebook

In the notebook, use **Option B** (data preparation):

```python
# OPTION B: Prepare data here
!python scripts/prepare_training_data.py \
    --config config/base_config.yaml \
    --output-dir /kaggle/working/nli_dataset \
    --balance-classes

DATA_DIR = '/kaggle/working/nli_dataset'
```

This will prepare data on Kaggle (~10 min).

### Step 3: Run Training

Same as Approach A, Step 5.

---

## 🔧 Troubleshooting

### Problem: "GPU quota exceeded"

**Solution**: You've used 30 hours this week. Options:
- Wait until next week (quota resets)
- Use Colab instead
- Upgrade to Kaggle Paid (if available)

### Problem: "Session timeout after 12 hours"

**Solution**: Training should finish in 7-8 hours. If it doesn't:
- Check if using T4 (not CPU)
- Verify `mixed_precision: fp16`
- Resume from checkpoint:
  ```python
  --resume-from /kaggle/working/checkpoints/checkpoint-epoch-3
  ```

### Problem: "Out of memory"

**Solution**:
- Reduce batch size: `--batch-size 4`
- Increase gradient accumulation: `--gradient-accumulation-steps 16`
- Effective batch stays 64

### Problem: "Dataset not found"

**Solution**:
- Verify dataset is "Added" in right panel
- Check path: `!ls /kaggle/input/`
- Update `kaggle_dataset_path` in notebook

### Problem: "Import error for transformers"

**Solution**:
- Re-run installation cell
- Restart kernel
- Check internet is ON

---

## 📊 Expected Results

### Training Progress

```
Epoch 1/5
  Train Loss: 0.4521
  Val Loss: 0.3241
  Val F1 (macro): 0.7123

Epoch 2/5
  Train Loss: 0.2841
  Val Loss: 0.2512
  Val F1 (macro): 0.8234

...

Epoch 5/5
  Train Loss: 0.1234
  Val Loss: 0.1876
  Val F1 (macro): 0.8712

Training complete!
Best model: Epoch 4, Val F1: 0.8745
```

### Final Test Metrics

```
📊 Test Set Performance:

  Accuracy:      0.8842
  F1 (macro):    0.8567
  F1 (weighted): 0.8801

  Per-class F1 scores:
    Entailment:     0.9123
    Neutral:        0.7845
    Contradiction:  0.8734
```

**✅ Success Criteria**:
- Accuracy > 85%
- F1 (macro) > 82%
- All class F1 > 75%

---

## 💰 Cost Summary

| Item | Cost |
|------|------|
| Kaggle GPU (30h/week) | **$0** |
| Training time (~8h) | **$0** |
| Data storage | **$0** |
| **Total** | **$0** ✅ |

---

## ⏱️ Time Breakdown

| Step | Time |
|------|------|
| Setup Kaggle account | 5 min |
| Prepare data locally | 10 min |
| Upload dataset | 5 min |
| Create notebook | 5 min |
| **Training** | **7-8 hours** |
| Evaluation | 10 min |
| Download model | 5 min |
| **Total** | **~8-9 hours** |

---

## 🎯 Post-Training

### Use Model in RAG Pipeline

1. **Verify model location**:
   ```bash
   ls models/hallucination_detector/
   ```

2. **Config already points to it**:
   ```yaml
   # config/base_config.yaml
   training:
     output:
       final_model_dir: ./models/hallucination_detector
   ```

3. **Use in code**:
   ```python
   from src.core.config_loader import load_config
   from src.rag.rag_pipeline import RAGPipeline

   config = load_config('config/base_config.yaml')
   rag = RAGPipeline.from_config(config)  # Auto-loads detector

   result = rag.query(
       "When did the Simpsons first air?",
       detect_hallucinations=True
   )

   print(f"Answer: {result['answer']}")
   print(f"Hallucination: {result['hallucination_detected']}")
   print(f"Score: {result['hallucination_score']:.2f}")
   ```

4. **Test inference**:
   ```bash
   python -c "
   from src.rag.hallucination_detector import HallucinationDetector

   detector = HallucinationDetector('models/hallucination_detector')

   result = detector.detect(
       premise='The Simpsons first aired in the 1980s',
       hypothesis='The Simpsons started in 1989'
   )

   print(f\"Result: {result['label']}\")
   print(f\"Confidence: {result['confidence']:.4f}\")
   "
   ```

---

## 📚 Additional Resources

- **Kaggle Docs**: https://www.kaggle.com/docs/notebooks
- **TensorBoard**: Monitor training in notebook
- **Checkpoints**: Resume interrupted training
- **Logs**: Check `/kaggle/working/training.log`

---

## ✅ Checklist

Before starting:
- [ ] Kaggle account created
- [ ] Phone verified (for GPU)
- [ ] Datasets prepared locally
- [ ] Dataset uploaded to Kaggle
- [ ] Notebook created
- [ ] GPU enabled (T4 or P100)
- [ ] Internet enabled

During training:
- [ ] Training started successfully
- [ ] GPU detected (nvidia-smi)
- [ ] Loss decreasing
- [ ] Checkpoints saving

After training:
- [ ] Model exported
- [ ] Files downloaded
- [ ] Model tested locally
- [ ] Integrated into RAG

---

## 🆘 Need Help?

1. **Check training log**: `!tail -100 training.log`
2. **Check GPU usage**: `!nvidia-smi`
3. **Verify dataset**: `!ls /kaggle/input/`
4. **Check errors**: Look at last cells in notebook

---

**🎉 Ready to train! Follow the steps above and you'll have your model in ~8 hours.**
