## A. Project Overview

This laboratory work conducts a **comparative analysis of three pre-trained CNN models** — MobileNetV2, InceptionV3, and EfficientNetB0 — trained on a custom dataset of **20 Philippine medicinal plant species**. Each model was evaluated using accuracy, loss, precision, recall, F1-score, confusion matrix, ROC curve, AUC score, and Grad-CAM explainability heatmaps.

The dataset contains **5,015 images** across 20 classes, split into 4,012 training images (80%) and 1,003 validation images (20%).

---

## B. Dataset

- **Total Images:** 5,015
- **Classes:** 20 Philippine medicinal plant species
- **Training Set:** 4,012 images
- **Validation Set:** 1,003 images
- **Image Size:** 224 × 224
- **Batch Size:** 32
- **Dataset Path:** `/content/drive/MyDrive/IMAGE DATA SET`

### Plant Species Included:
ALOE VERA, ALUGBATI, AMPALAYA, ANISE, BANANA, CALAMANSI, CLOVE, GINSENG, GUAVA, GUYABANO, HILBAS, IPIL IPIL, KANGKONG, LEMON GRASS, MALUNGGAY, MANGO, OREGANO, PANDAN, PEPPERMINT, SANTOL

---

## C. Models Used

All three models were loaded with **ImageNet pre-trained weights** and fine-tuned using **transfer learning** (base layers frozen).

**Shared Custom Head Architecture:**
```
Rescaling(1./255) → BaseModel (frozen) → GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(20)
```

**Optimizer:** Adam (learning_rate = 0.0001)  
**Loss Function:** SparseCategoricalCrossentropy (from_logits=True)  
**Epochs:** 10

| Model | Total Params | Trainable Params | Base Params (Frozen) |
|-------|-------------|------------------|----------------------|
| MobileNetV2 | 2,424,532 | 166,548 | 2,257,984 |
| InceptionV3 | 22,067,636 | 264,852 | 21,802,784 |
| EfficientNetB0 | 4,216,119 | 166,548 | 4,049,571 |

---

## D. Training History (Epoch-by-Epoch)

### MobileNetV2
| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 0.1588 | 2.8629 | 0.5304 | 2.1107 |
| 2 | 0.4339 | 1.9778 | 0.7228 | 1.3617 |
| 3 | 0.6037 | 1.4527 | 0.7966 | 0.9787 |
| 4 | 0.6844 | 1.1598 | 0.8305 | 0.7954 |
| 5 | 0.7276 | 0.9925 | 0.8475 | 0.6747 |
| 6 | 0.7724 | 0.8371 | 0.8634 | 0.5924 |
| 7 | 0.8016 | 0.7595 | 0.8734 | 0.5345 |
| 8 | 0.8108 | 0.6952 | 0.8933 | 0.4847 |
| 9 | 0.8362 | 0.6124 | 0.8973 | 0.4474 |
| **10** | **0.8470** | **0.5745** | **0.9043** | **0.4135** |

### InceptionV3
| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 0.2109 | 2.6606 | 0.6002 | 1.8940 |
| 2 | 0.4925 | 1.7769 | 0.7109 | 1.2727 |
| 3 | 0.6174 | 1.3561 | 0.7767 | 0.9746 |
| 4 | 0.6894 | 1.1184 | 0.8126 | 0.8178 |
| 5 | 0.7338 | 0.9593 | 0.8445 | 0.7083 |
| 6 | 0.7724 | 0.8360 | 0.8574 | 0.6353 |
| 7 | 0.7996 | 0.7256 | 0.8744 | 0.5696 |
| 8 | 0.8238 | 0.6665 | 0.8853 | 0.5295 |
| 9 | 0.8427 | 0.5979 | 0.8923 | 0.4886 |
| **10** | **0.8524** | **0.5563** | **0.9003** | **0.4467** |

### EfficientNetB0
| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 0.0503 | 3.0145 | 0.0548 | 2.9958 |
| 2 | 0.0523 | 3.0014 | 0.0449 | 2.9986 |
| 3 | 0.0603 | 2.9972 | 0.0449 | 2.9966 |
| 4 | 0.0546 | 2.9964 | 0.0449 | 2.9966 |
| 5 | 0.0518 | 2.9969 | 0.0499 | 2.9983 |
| 6 | 0.0496 | 2.9967 | 0.0449 | 2.9960 |
| 7 | 0.0528 | 2.9959 | 0.0518 | 2.9961 |
| 8 | 0.0516 | 2.9959 | 0.0359 | 2.9961 |
| 9 | 0.0516 | 2.9959 | 0.0359 | 2.9961 |
| **10** | **0.0536** | **2.9956** | **0.0359** | **2.9962** |

---

## E. Per-Class Classification Report

### MobileNetV2 — Per-Class Metrics
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ALOE VERA | 0.81 | 0.93 | 0.86 | 41 |
| ALUGBATI | 0.84 | 0.98 | 0.91 | 49 |
| AMPALAYA | 0.91 | 0.93 | 0.92 | 45 |
| ANISE | 0.92 | 1.00 | 0.96 | 45 |
| BANANA | 0.85 | 0.97 | 0.91 | 36 |
| CALAMANSI | 0.87 | 0.82 | 0.85 | 50 |
| CLOVE | 0.90 | 0.89 | 0.90 | 53 |
| GINSENG | 0.90 | 0.84 | 0.87 | 62 |
| GUAVA | 0.81 | 0.88 | 0.84 | 48 |
| GUYABANO | 0.98 | 0.90 | 0.94 | 52 |
| HILBAS | 0.83 | 1.00 | 0.91 | 50 |
| IPIL IPIL | 1.00 | 0.89 | 0.94 | 46 |
| KANGKONG | 0.92 | 0.85 | 0.88 | 52 |
| LEMON GRASS | 0.96 | 0.94 | 0.95 | 54 |
| MALUNGGAY | 0.92 | 0.87 | 0.90 | 55 |
| MANGO | 0.92 | 0.92 | 0.92 | 51 |
| OREGANO | 0.92 | 0.92 | 0.92 | 52 |
| PANDAN | 0.91 | 0.90 | 0.91 | 48 |
| PEPPERMINT | 0.98 | 0.89 | 0.93 | 55 |
| SANTOL | 0.94 | 0.83 | 0.88 | 59 |
| **Weighted Avg** | **0.91** | **0.90** | **0.90** | **1003** |

### InceptionV3 — Per-Class Metrics
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ALOE VERA | 0.84 | 0.93 | 0.88 | 41 |
| ALUGBATI | 0.85 | 0.94 | 0.89 | 49 |
| AMPALAYA | 0.96 | 0.96 | 0.96 | 45 |
| ANISE | 1.00 | 1.00 | 1.00 | 45 |
| BANANA | 0.82 | 0.86 | 0.84 | 36 |
| CALAMANSI | 0.84 | 0.74 | 0.79 | 50 |
| CLOVE | 0.89 | 0.92 | 0.91 | 53 |
| GINSENG | 0.91 | 0.84 | 0.87 | 62 |
| GUAVA | 0.80 | 0.92 | 0.85 | 48 |
| GUYABANO | 0.96 | 0.96 | 0.96 | 52 |
| HILBAS | 0.92 | 0.94 | 0.93 | 50 |
| IPIL IPIL | 0.80 | 0.87 | 0.83 | 46 |
| KANGKONG | 0.92 | 0.94 | 0.93 | 52 |
| LEMON GRASS | 0.96 | 0.94 | 0.95 | 54 |
| MALUNGGAY | 0.84 | 0.85 | 0.85 | 55 |
| MANGO | 0.94 | 0.88 | 0.91 | 51 |
| OREGANO | 0.96 | 0.92 | 0.94 | 52 |
| PANDAN | 0.83 | 0.92 | 0.87 | 48 |
| PEPPERMINT | 0.98 | 0.87 | 0.92 | 55 |
| SANTOL | 0.98 | 0.83 | 0.90 | 59 |
| **Weighted Avg** | **0.90** | **0.90** | **0.90** | **1003** |

### EfficientNetB0 — Per-Class Metrics
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ALOE VERA | 0.00 | 0.00 | 0.00 | 41 |
| ALUGBATI | 0.00 | 0.00 | 0.00 | 49 |
| AMPALAYA | 0.00 | 0.00 | 0.00 | 45 |
| ANISE | 0.00 | 0.00 | 0.00 | 45 |
| BANANA | 0.04 | 1.00 | 0.07 | 36 |
| CALAMANSI | 0.00 | 0.00 | 0.00 | 50 |
| CLOVE | 0.00 | 0.00 | 0.00 | 53 |
| GINSENG | 0.00 | 0.00 | 0.00 | 62 |
| GUAVA | 0.00 | 0.00 | 0.00 | 48 |
| GUYABANO | 0.00 | 0.00 | 0.00 | 52 |
| HILBAS | 0.00 | 0.00 | 0.00 | 50 |
| IPIL IPIL | 0.00 | 0.00 | 0.00 | 46 |
| KANGKONG | 0.00 | 0.00 | 0.00 | 52 |
| LEMON GRASS | 0.00 | 0.00 | 0.00 | 54 |
| MALUNGGAY | 0.00 | 0.00 | 0.00 | 55 |
| MANGO | 0.00 | 0.00 | 0.00 | 51 |
| OREGANO | 0.00 | 0.00 | 0.00 | 52 |
| PANDAN | 0.00 | 0.00 | 0.00 | 48 |
| PEPPERMINT | 0.00 | 0.00 | 0.00 | 55 |
| SANTOL | 0.00 | 0.00 | 0.00 | 59 |
| **Weighted Avg** | **0.00** | **0.04** | **0.00** | **1003** |

> ⚠️ EfficientNetB0 failed to learn — it predicted only BANANA for all 1,003 validation inputs. This was caused by a **preprocessing mismatch**: EfficientNetB0 has built-in normalization and should NOT have an additional `Rescaling(1./255)` layer prepended. The double-normalization made inputs unrecognizable to the model.

---

## F. Overall Performance Comparison Table

| Model | Train Acc | Train Loss | Test Acc | Test Loss | Precision | Recall | F1-Score | AUC |
|-------|-----------|------------|----------|-----------|-----------|--------|----------|-----|
| **MobileNetV2** | **0.8470** | **0.5745** | **0.9043** | **0.4135** | **0.9081** | **0.9043** | **0.9043** | **0.9951** |
| InceptionV3 | 0.8524 | 0.5563 | 0.9003 | 0.4467 | 0.9036 | 0.9003 | 0.9005 | 0.9937 |
| EfficientNetB0 | 0.0536 | 2.9956 | 0.0359 | 2.9962 | 0.0013 | 0.0359 | 0.0025 | 0.5001 |

> ✅ **Best Model: MobileNetV2** — Highest test accuracy (90.43%), highest AUC (0.9951), and highest weighted F1-score (0.9043).

---

## G. Comparison with Previous Laboratory Work Models

From **LW4 (EDERIO_LW4.ipynb)**, the Teachable Machine model and the custom CNN enhancement were also evaluated on the same 20-class, 1,003-image validation set:

| Model | Source | Test Accuracy | Precision (W) | Recall (W) | F1-Score (W) | AUC |
|-------|--------|--------------|---------------|------------|--------------|-----|
| Teachable Machine | LW4 (loaded) | 0.6400 | 0.67 | 0.64 | 0.64 | 0.9256 |
| Custom CNN (20 epochs) | LW4 (enhanced) | 0.4277 | — | — | — | — |
| **MobileNetV2** | **LW5** | **0.9043** | **0.9081** | **0.9043** | **0.9043** | **0.9951** |
| InceptionV3 | LW5 | 0.9003 | 0.9036 | 0.9003 | 0.9005 | 0.9937 |
| EfficientNetB0 | LW5 | 0.0359 | 0.0013 | 0.0359 | 0.0025 | 0.5001 |

Transfer learning (MobileNetV2 and InceptionV3) significantly outperformed the custom-built CNNs from LW4, confirming the power of pre-trained ImageNet features for plant classification tasks.

---

## H. Guide Questions — Final Reflection

### A. Model Performance

**1. Which pre-trained model achieved the highest accuracy? Why?**

MobileNetV2 achieved the highest test accuracy at **90.43%**, narrowly outperforming InceptionV3 (90.03%) and vastly surpassing EfficientNetB0 (3.59%). MobileNetV2 performed best in this setup because its depthwise separable convolution architecture extracts generalizable features that transferred well to plant classification without requiring fine-tuning. It also converged faster — reaching over 85% validation accuracy by epoch 7 — and benefited from stable, consistent improvement across all 10 epochs with no signs of instability or plateau.

**2. Which model had the lowest performance? What could be the reason?**

EfficientNetB0 had the lowest performance by a huge margin, achieving only **3.59% test accuracy** — barely above the 5% expected from random guessing on a 20-class problem. The root cause was a **preprocessing mismatch**: the pipeline prepended a `Rescaling(1./255)` layer before EfficientNetB0, but EfficientNetB0 already applies its own internal normalization. This double-normalization scaled pixel values far below the range the model expected, making all inputs unrecognizable. The model collapsed to always predicting the BANANA class, which happened to be alphabetically and positionally close to the default logit peak.

**3. How did loss values compare across models?**

MobileNetV2 ended with the lowest validation loss at **0.4135**, followed by InceptionV3 at **0.4467**. Both showed healthy, steady decreases in loss across all 10 epochs with training and validation loss tracking closely — a sign of good generalization without significant overfitting. EfficientNetB0 maintained a near-constant loss of approximately **2.9960** throughout all epochs. This flat loss curve indicates the model's gradients were effectively zero — it was receiving no useful learning signal because its inputs were completely outside the distribution it was designed for.

---

### B. Evaluation Metrics

**4. Why is accuracy not enough to evaluate a model?**

Accuracy alone is misleading in multi-class classification problems. A clear example from this experiment is EfficientNetB0 — it achieved 3.59% accuracy while having 100% recall on the BANANA class but 0% recall on all other 19 classes. Looking only at overall accuracy, one might assume the model is simply weak, but the per-class metrics reveal it completely failed to learn any distinguishing features. In a medicinal plant application, a model that consistently misidentifies one plant as another could lead to incorrect medicinal use, making precision, recall, and F1-score per class far more informative than a single accuracy number.

**5. Which model had the best F1-score? What does it indicate?**

MobileNetV2 had the best weighted F1-score of **0.9043**. This indicates that it maintained a strong and consistent balance between precision (avoiding false positives) and recall (avoiding false negatives) across all 20 plant classes. Its highest individual F1-scores were for ANISE (0.96), LEMON GRASS (0.95), GUYABANO (0.94), and IPIL IPIL (0.94) — all species with highly distinct visual features. The weighted F1-score of 0.9043 means this performance held up even for harder-to-distinguish classes.

**6. How did Precision and Recall differ across models?**

For MobileNetV2, precision ranged from 0.81 (ALOE VERA, GUAVA) to 1.00 (IPIL IPIL), and recall ranged from 0.82 (CALAMANSI) to 1.00 (ANISE, HILBAS). The model was slightly more consistent in recall than precision, suggesting it was better at finding true positives than avoiding false positives. For InceptionV3, CALAMANSI had notably low recall (0.74) and BANANA had low recall (0.86), which pulled its weighted average slightly below MobileNetV2. EfficientNetB0 scored 0.00 on both precision and recall for 19 of 20 classes, with the only exception being BANANA — recall of 1.00 but precision of only 0.04, since it classified everything as BANANA.

---

### C. Confusion Matrix Analysis

**7. Which classes were frequently misclassified?**

Based on the per-class metrics, the most frequently misclassified classes were:
- **CALAMANSI** — recall of 0.82 in MobileNetV2 and 0.74 in InceptionV3, likely confused with MANGO or GUAVA due to similar round, smooth-edged leaves
- **GINSENG** — recall of 0.84 in both models, likely misidentified as MALUNGGAY since both have compound, multi-leaflet structures
- **SANTOL** — recall of 0.83 in MobileNetV2 and 0.83 in InceptionV3, frequently confused with GUAVA due to overlapping leaf morphology
- **GUAVA** — low precision (0.81 in MobileNetV2, 0.80 in InceptionV3), meaning other species were often incorrectly predicted as GUAVA

**8. What patterns did you observe in the confusion matrix?**

Both MobileNetV2 and InceptionV3 confusion matrices showed strong diagonal dominance, confirming high overall classification accuracy. Off-diagonal errors clustered around visually similar species: GUAVA/SANTOL, CALAMANSI/MANGO, and GINSENG/MALUNGGAY were the most common confusion pairs, all sharing similar leaf shape, color, and texture. Classes with structurally unique morphology — ANISE (feathery, lacy leaves), LEMON GRASS (long narrow blade), PANDAN (elongated strappy leaves), and ALUGBATI (round succulent leaves) — showed very high diagonal values and minimal confusion. EfficientNetB0's confusion matrix showed an entirely blank off-diagonal with one completely filled column (BANANA), confirming the single-class collapse.

---

### D. ROC and AUC

**9. Which model had the highest AUC score?**

MobileNetV2 had the highest mean AUC of **0.9951**, followed by InceptionV3 at **0.9937**. EfficientNetB0 scored **0.5001**, equivalent to random guessing — it had no ability to separate any class from any other. The AUC gap between MobileNetV2 and InceptionV3 (0.0014) is small but consistent with MobileNetV2's slightly better precision across most classes.

**10. What does AUC tell us about model performance?**

AUC (Area Under the ROC Curve) measures a model's ability to distinguish between classes regardless of the chosen classification threshold. A score of 1.0 means perfect separability; 0.5 means random guessing. For a 20-class problem, a mean AUC of 0.9951 (MobileNetV2) means that across all class-vs-all comparisons, the model correctly ranked a true positive higher than a false positive in 99.51% of cases. AUC is threshold-independent and is more informative than accuracy because it evaluates the model's entire scoring behavior, not just its final decision at the 0.5 cutoff.

---

### E. Explainability (Grad-CAM)

**11. What did Grad-CAM reveal about model decision-making?**

Grad-CAM was successfully applied to MobileNetV2 (last conv layer: `Conv_1`) and EfficientNetB0 (last conv layer: `top_conv`). For MobileNetV2, heatmaps showed activation concentrated on the leaf blade — particularly around distinctive edges, vein patterns, and surface textures — confirming that the model learned genuine plant-specific features. InceptionV3's Grad-CAM encountered a shape incompatibility error (`expected axis -1 to have value 2048, received shape (1, 192)`), caused by an intermediate conv layer returning a reduced spatial output that was incompatible with the dense head, so no valid heatmaps were produced for InceptionV3.

**12. Did the model focus on relevant image regions?**

For MobileNetV2, yes — heatmaps consistently highlighted the leaf surface and boundary regions, with the strongest activations over distinctive structural features such as serrated edges (GUAVA, SANTOL), elongated shape (LEMON GRASS, PANDAN), and dense texture (OREGANO, PEPPERMINT). This confirms that MobileNetV2's predictions were grounded in the actual visual properties of each plant. For EfficientNetB0, heatmaps were generated but were meaningless — since the model never learned to distinguish classes, the activations were random noise across the image rather than focused plant regions.

**13. Which model produced the most meaningful heatmaps?**

MobileNetV2 produced the most meaningful and interpretable Grad-CAM heatmaps. Its `Conv_1` layer was successfully identified and used to generate focused activations that highlighted plant-specific regions. InceptionV3 failed to produce any heatmaps due to the dimensional mismatch error during the gradient computation. EfficientNetB0 technically generated heatmaps from `top_conv`, but since the model predicted only one class for all inputs, the heatmaps did not reflect any meaningful learned features.

---

### F. Model Comparison & Improvement

**14. Which model would you recommend for deployment? Why?**

**MobileNetV2** is the recommended model for deployment. It achieved the best results across all metrics — highest test accuracy (90.43%), highest AUC (0.9951), and highest F1-score (0.9043). Beyond performance, MobileNetV2 is architecturally designed for mobile and edge deployment: at just 2.4M total parameters, it is the lightest and fastest of the three models, making it ideal for running on smartphones without internet connectivity. Its Grad-CAM heatmaps were also the most interpretable, which matters for building user trust in a healthcare-adjacent application where users need confidence in the model's reasoning.

**15. How can you further improve your best-performing model?**

Several concrete improvements could push MobileNetV2 and the overall pipeline further:
- **Fix EfficientNetB0**: Remove the `Rescaling(1./255)` layer before EfficientNetB0, since it uses internal preprocessing. With this fix, EfficientNetB0 could potentially outperform MobileNetV2 given its larger parameter count.
- **Fine-tuning**: Unfreeze the top 30–50 layers of MobileNetV2 after the initial 10 epochs and continue training at a lower learning rate (1e-5) to adapt ImageNet features more specifically to Philippine medicinal plant morphology.
- **Data augmentation**: Apply random rotation, zoom, horizontal flip, and brightness/contrast jitter to reduce sensitivity to lighting and angle variation in real-world conditions.
- **Address weak classes**: Collect additional high-quality images for CALAMANSI, GINSENG, SANTOL, and GUAVA — the consistently low-performing classes — to reduce confusion between visually similar species.
- **Learning rate scheduling**: Use `ReduceLROnPlateau` or cosine annealing to automatically lower the learning rate when validation loss stagnates, squeezing more accuracy from additional epochs.

---

### G. Real-World Application

**16. How can your model be applied in real-world scenarios?**

The trained MobileNetV2 model can be deployed as a **mobile plant identification application** for farmers, students, barangay health workers, and community members across the Philippines. A user simply photographs an unknown plant and receives an instant identification paired with the plant's medicinal uses, scientific name, local names, and preparation instructions. Beyond individual use, the system could be integrated into herbalism and traditional medicine training programs, biodiversity monitoring for Philippine flora, and agricultural extension services that help communities identify and cultivate medicinal crops. It also has educational potential as an interactive botany learning tool for school students.

**17. What are the risks of deploying an inaccurate model?**

The most serious risk is **misidentification leading to harmful consumption** — if a toxic plant is mistakenly identified as a medicinal one (e.g., misidentifying a look-alike of MALUNGGAY), a user could self-medicate with dangerous substances. Overconfident model outputs with no uncertainty estimate could cause users to bypass professional verification. Class-specific failures — such as consistently confusing CALAMANSI with MANGO — while not toxic, could lead to incorrect remedies being used. To mitigate these risks, the deployed application should always display a confidence score alongside each prediction, issue a clear warning (e.g., "Low confidence — please verify") for predictions below 80%, and prominently recommend consulting a licensed herbalist or medical professional before using any plant for medicinal purposes.

**18. How can this system be integrated into a mobile/web app?**

The model can be exported in **TensorFlow Lite** (`.tflite`) format for efficient on-device inference on Android and iOS without requiring an internet connection. A **Flutter** frontend can capture images from the device camera, preprocess them to 224×224, run the TFLite model locally, and display the top-3 predictions with confidence scores and plant descriptions in both English and Filipino. Alternatively, the model can be served as a **REST API** using FastAPI or Flask deployed on Google Cloud Run or AWS Lambda, with a simple HTML/React web frontend that accepts image uploads and returns predictions. For offline-first community use, the TFLite approach is recommended given limited connectivity in rural areas of the Philippines where medicinal plant knowledge is most actively practiced.

---

## I. GitHub Repository Structure

```
Plant-Species-Image-Classification/
├── EderioLW5.ipynb              # LW5 — Comparative analysis of 3 pre-trained CNNs
├── EDERIO_LW4.ipynb             # LW4 — Custom CNN + Grad-CAM + evaluation
├── EDERIO_LW3_CSC120.ipynb      # LW3 — Initial model training
├── README.md                    # This file
├── Image/                       # Per-class training result screenshots
└── images/
    └── plants/                  # Test screenshots, confusion matrix, accuracy charts
```

---

## Project Links

- **Google Colab Notebook:** [(https://colab.research.google.com/drive/1M--KYuHoySq-LMp6UYaaxvIOw3_OPY-U?usp=sharing)]
  
