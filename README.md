# PolyNet: Data-driven Framework for Property Prediction of Polymers

<img width="955" height="295" alt="Screen Shot 2025-08-26 at 12 05 13" src="https://github.com/user-attachments/assets/08c2c626-c9d9-48b3-bc8d-d4131625f66e" />

## Abstract
PolyNet is a structure–property prediction framework for soft-matter systems such as polymers and hydrogels. It combines sequence semantic representations from the pretrained molecular language model polyBERT with experimentally measurable macroscopic material features (water content, swelling ratio, degree of polymerization, mechanical parameters, etc.) to build an extensible fused regression system. The framework provides multiple lightweight model heads (Light / Residual / Attention / Ensemble / CNN1D), supports command-line training and inference, batch prediction, and interactive Notebook analysis. It is suitable for rapid modeling and comparative validation on small-to-moderate experimental datasets.

## 1. Motivation
- Polymer material structures are diverse and hierarchically complex; purely empirical rules or simple molecular fingerprints cannot fully represent them.
- The pretrained molecular Transformer (polyBERT) can capture sequence-level semantics and local environment patterns.
- Experimental material features (macroscopic / processing / morphology / physical-statistical metrics) are complementary to molecular-level semantics.
- By feature fusion and multiple pluggable regression heads, the framework achieves performance–generalization trade-offs at different levels of complexity.

## 2. Method Overview
1. SMILES encoding: use kuelumbus/polyBERT; tokenize → forward through Transformer → mean pooling to obtain a fixed-dimension vector.
2. Material feature processing: optional standardization (StandardScaler).
3. Fusion mechanism: project material features to a unified fusion dimension and interact them with the compressed polyBERT representation (attention / concatenation / residual).
4. Model heads：
   - Light：compact MLP
   - Residual：stacked residual blocks
   - Attention：self-attention + feature-importance weighting
   - Ensemble：multi-branch candidates + Softmax-weighted aggregation
   - CNN1D：treat the fused vector as a 1D sequence and extract local patterns via convolutions and pooling
5. Optimization: mean squared error (MSE) as the primary loss; report MAE / RMSE / R².
6. Saving: model weights (state_dict) and feature scaler (feature_scaler.pkl, if normalization is enabled), with configuration and training-curve files.

## 3. Directory Structure
```
polyNet/
  model_training.py
  model_prediction.py
  polyBERT.ipynb
  trained_models/
    attention_20C/
      model.pth
      feature_scaler.pkl
  datasets/
  requirements.txt
  README.md
```

## 4. Environment Setup
```bash
python -m venv polyNet
source venv/bin/activate  
pip install -r requirements.txt
```

## 5. Data Format
Minimum required fields:
- SMILES: molecular structure
- Target column: e.g., Conductivity (can be replaced by any desired property: ionic conductivity, tensile strength, dielectric constant, etc.)
- Material features: any numeric columns (examples: WaterContent, SwellingRate, Degreeofpolymerization, ElongationatBreak, TensileStrength)

CSV/Excel row example:
```
SMILES,WaterContent,SwellingRate,Degreeofpolymerization,ElongationatBreak,TensileStrength,Conductivity
C=CO,87.5,146.4,1750,171.9,0.16,0.012
C(CO)O,90.0,143.9,100,209.3,0.16,0.018
```


## 6. Training Command-line Usage
Minimal example (structure + target only):
```bash
python model_training.py \
  --data_path datasets/20C_dataset.xlsx \
  --target_col Conductivity
```

Full example:
```bash
python model_training.py \
  --data_path datasets/20C_dataset.xlsx \
  --target_col Conductivity \
  --features WaterContent,SwellingRate,Degreeofpolymerization,ElongationatBreak,TensileStrength \
  --model_type attention \
  --epochs 40 \
  --batch_size 2 \
  --lr 5e-4 \
  --weight_decay 1e-5 \
  --lr_step_size 50 \
  --lr_gamma 0.5 \
  --test_size 2 \
  --save_dir trained_models/attention_20C_run1 \
  --seed 42
```

Parameter descriptions:
| Parameter | Description |
|------|------|
| --data_path | Path to data file (Excel/CSV) |
| --target_col | Target property column name |
| --features | Comma-separated material numeric feature columns |
| --model_type | light / residual / attention / ensemble / cnn1d |
| --epochs / --batch_size | Training epochs / batch size |
| --lr / --weight_decay | Learning rate / L2 regularization |
| --lr_step_size / --lr_gamma | Learning rate scheduler |
| --test_size | <1 means fraction; ≥1 means fixed number of test samples |
| --no_normalize_features | Disable material feature standardization |
| --save_dir | Model output directory |
| --seed | Random seed |


## 7. Inference and Deployment
Single prediction:
```bash
python model_prediction.py \
  --model trained_models/attention_20C_run1/model.pth \
  --smiles "C(=O)(O)C=C"
```

Batch prediction:
```bash
python model_prediction.py \
  --model trained_models/attention_20C_run1/model.pth \
  --csv datasets/new_samples.csv \
  --smiles_col SMILES \
  --output datasets/new_samples_predictions.csv
```

Interactive mode (one-by-one input):
```bash
python model_prediction.py -i
```

Notebook usage example:
```python
from model_prediction import ModelPredictor
predictor = ModelPredictor("trained_models/attention_20C_run1/model.pth")
y = predictor.predict_single(
    smiles="C(=O)(O)C=C",
    material_properties={
        "WaterContent": 89.3,
        "SwellingRate": 143.6,
        "Degreeofpolymerization": 1750,
        "ElongationatBreak": 190.7,
        "TensileStrength": 0.75
    }
)
print("Pred:", y)
```

Batch:
```python
preds = predictor.predict_batch(
    smiles_list=["C=CO","C(CO)O"],
    material_properties_list=[
        {"WaterContent":87.5,"SwellingRate":146.4,"Degreeofpolymerization":1750,"ElongationatBreak":171.9,"TensileStrength":0.16},
        {"WaterContent":90.0,"SwellingRate":143.9,"Degreeofpolymerization":100,"ElongationatBreak":209.3,"TensileStrength":0.16}
    ]
)
```

## 8. Model Architecture
| Component | Role |
|------|------|
| polyBERT Encoder | Pretrained sequence representation that captures atom-sequence semantics and local environments |
| Mean Pooling | Token-level representations → global molecular vector |
| Feature Projector | Map material features to fusion dimension (stacked linear layers + activation + dropout) |
| Cross / Self Attention | Capture multimodal interactions and internal correlations |
| Ensemble Branches | Multiple sub-model outputs aggregated with weights to reduce single-model variance |
| CNN1D Head | Convolutional extraction of local response patterns to improve nonlinear modeling capability |

Possible extensions: multi-task objectives, graph-structure feature fusion, contrastive fine-tuning, parameter-efficient fine-tuning (LoRA / Adapters).



