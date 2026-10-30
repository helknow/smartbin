# SmartBin — AI Waste-Sorting Classifier (Design Spec)

**Date:** 2026-10-30
**Course:** ICT, Assignment 2 (group project defence, Week 10, 09–14 Nov 2026)
**Area:** Computer Vision
**Status:** Draft, awaiting review

---

## 1. Goal and success criteria

**Problem.** People often put recyclables in the wrong bin, and sorting them by hand is slow. SmartBin takes a photo of one waste item and says which bin it belongs in.

**What the assignment asks for (from the instructor's email):**
- Requirement **option 1**: use an open-source dataset and change the ML algorithm (layers, hyperparameters, optimisers) and compare the results.
- An interface that makes the project look like an MVP (an advantage).
- A presentation of up to 10 minutes in Week 10.

**Done means:**
1. At least 4 training experiments on the same data split, each changing **one** thing compared with the previous one.
2. One comparison table and chart: test accuracy, macro F1, training time and trainable parameters for each experiment.
3. The best model reaches **≥ 85 % test accuracy** on TrashNet. Published VGG16 transfer-learning results on this dataset are around 85–93 %.
4. A working Gradio app. You upload a photo and see the bin, a confidence score and the top 3 classes.
5. Slides and a live demo that fit in 10 minutes.

**What the instructor said vs. our assumptions:**
- *Said:* open-source dataset, change the algorithm, compare, interface is a bonus, 10-minute defence, groups of up to 6.
- *Assumed:* PyTorch and torchvision (the email shows a torchvision VGG16 printout); free Google Colab GPU for training; team size of 6 (the plan works with fewer people by merging roles).

## 2. Scope

**In scope:** image classification of one item per photo into 6 classes, transfer learning from pretrained VGG16, experiments, evaluation, Gradio app, README. Optional: MobileNetV2 comparison, Dockerfile, Hugging Face Spaces deploy.

**Out of scope:** collecting our own photos (that is option 2), object detection of several items in one photo, mobile app, real-time video.

## 3. Dataset

- **TrashNet** (Gary Thung & Mindy Yang, Stanford), `dataset-resized.zip` from `github.com/garythung/trashnet`. The Hugging Face mirror `garythung/trashnet` is a fallback.
- 2,527 RGB photos, 512×384, one item on a white background.
- Classes and counts: cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137.
- **Imbalance:** the `trash` class is small. We report **macro F1** next to accuracy so a weak `trash` score is not hidden.
- **Split:** stratified 70 / 15 / 15 (train / val / test), fixed seed `42`, saved to `splits.csv` so every experiment uses the same images.
- **Preprocessing (the same for every experiment):**
  - Train: `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip`, `ToTensor`, ImageNet normalisation.
  - Val / test / app: `Resize(256)`, `CenterCrop(224)`, `ToTensor`, ImageNet normalisation.

## 4. Experiments

All runs use pretrained `torchvision.models.vgg16(weights="IMAGENET1K_V1")`, batch size 32, 10 epochs, and cross-entropy loss. The checkpoint with the best **validation** accuracy is kept. The **test** set is evaluated once per experiment, at the end.

| ID | What changes | Features (conv) | Classifier head | Optimiser |
|----|---------------|-----------------|-----------------|-----------|
| E1 | Baseline | frozen | Original VGG16 head, last layer `Linear(4096→6)` | SGD, lr 1e-3, momentum 0.9 |
| E2 | Smaller head | frozen | `Linear(25088→1024) → ReLU → Dropout(0.3) → Linear(1024→6)` | SGD, lr 1e-3, momentum 0.9 |
| E3 | Optimiser | frozen | as E2 | Adam, lr 1e-4 |
| E4 | Fine-tuning | last conv block unfrozen (`features[24:]`) | as E2 | Adam: conv lr 1e-5, head lr 1e-4 |
| E5 *(optional)* | Different architecture | MobileNetV2, frozen | `Dropout(0.2) → Linear(1280→6)` | Adam, lr 1e-4 |

**How to read the results in the defence:** E1→E2 shows the effect of the head size (fewer parameters, faster training). E2→E3 shows the effect of the optimiser. E3→E4 shows the effect of fine-tuning. E5 shows the trade-off between speed and accuracy.

**Logged for each run:** train and val loss and accuracy per epoch, test accuracy, macro F1, per-class precision, recall and F1, confusion matrix, total training time, and the number of trainable parameters.

## 5. Project structure

```
smartbin/
├── notebooks/
│   └── smartbin_training.ipynb   # runs in Colab: data → experiments → results
├── model.py                      # single source of truth: CLASSES, transforms, build_model(exp_id)
├── app.py                        # Gradio interface
├── tests/
│   └── test_model.py             # small pytest checks
├── results/
│   ├── results.csv               # one row per experiment
│   ├── comparison.png            # bar chart: accuracy + macro F1 per experiment
│   ├── curves_E*.png             # loss and accuracy curves
│   └── confusion_E*.png          # confusion matrices
├── weights/
│   └── best.pt                   # best checkpoint (not committed if > 100 MB)
├── requirements.txt
├── Dockerfile                    # optional
└── README.md
```

**Units and their contracts:**

- **`model.py`**
  - `CLASSES`: the 6 class names in a fixed order.
  - `BIN_INFO`: for each class, the bin name, a colour and a one-line disposal tip.
  - `train_transform()` and `eval_transform()`.
  - `build_model(exp_id: str) -> nn.Module` builds the E1–E5 architectures and sets which layers are frozen.
  - Depends only on torch and torchvision. The notebook writes this file with `%%writefile model.py`, so the training code and the app always use the same definitions.

- **Notebook** (cells in order): setup → download and split the data → `train_one(exp_id)` loop → evaluate → save `results.csv`, charts and `best.pt`.
  - The checkpoint format is `{"exp_id": str, "classes": list, "state_dict": dict, "test_acc": float}`, so the app can rebuild the right architecture.

- **`app.py`**
  - `load(path) -> nn.Module` reads the checkpoint and calls `build_model(ckpt["exp_id"])`.
  - `predict(image: PIL.Image) -> dict` returns the top 3 classes with probabilities, the bin name and the disposal tip.
  - The Gradio `Blocks` UI has an image upload, the top-3 label output, a bin card (name, colour, tip), example images (one per class) and a small "About the model" section that shows the comparison chart.

## 6. Data flow

```
TrashNet zip → split (seed 42, splits.csv) → DataLoaders → train_one(E1..E5)
  → best checkpoint per experiment (by val acc) → test evaluation → results.csv + charts
  → best overall → weights/best.pt → app.py → user photo → eval_transform → model → softmax → top-3 + bin card
```

## 7. Error handling (app)

- Non-RGB images (RGBA, grayscale, palette) are converted to RGB before the transform.
- If no image is uploaded, the app shows a prompt instead of an error.
- If the top probability is below **0.5**, the app shows "Not sure. Try a clearer photo of one item on a plain background" next to the top-3 list.
- If `weights/best.pt` is missing, the app stops at startup with a clear message telling the user to run the notebook first.
- The app runs on CPU by default and uses CUDA when it is available.

## 8. Testing

- `tests/test_model.py` (pytest, CPU, no weights needed):
  - For every `exp_id`, `build_model(exp_id)(torch.randn(2, 3, 224, 224))` has shape `(2, 6)`.
  - Frozen layers have `requires_grad == False`, and the expected layers are trainable.
  - `predict()` on a random RGB image returns 3 classes whose probabilities are in [0, 1], and the full softmax sums to 1. This uses a randomly initialised model.
  - `predict()` accepts an RGBA image and a grayscale image.
- Sanity check in the notebook: before the real runs, E2 must overfit 1 batch (train accuracy → 100 %) to prove the training loop works.
- Manual app check: one example photo per class, plus one photo that is not waste (a low-confidence message is expected).

## 9. Deliverables for the defence

1. Repository or zip with the notebook, `model.py`, `app.py`, results and README.
2. Slides (about 8), roughly one minute each:
   problem → dataset → VGG16 and what we changed → experiment table → results chart → confusion matrix of the best model → live demo → conclusions and next steps.
3. Live Gradio demo. Recorded video as a backup in case the Wi-Fi fails.

## 10. Team split (6 people)

> **Update (2026-10-30):** this is a solo project. The roles below are one person's checklist, and the 6-person assumption in section 1 no longer applies.

| Role | Owns |
|------|------|
| 1. Data | Download, split, `splits.csv`, sample-image grid |
| 2. Experiments A | E1, E2, training loop, overfit check |
| 3. Experiments B | E3, E4 (and E5 if time) |
| 4. Evaluation | `results.csv`, charts, confusion matrices, comparison analysis |
| 5. App | `app.py`, `BIN_INFO`, tests, optional Dockerfile and HF Spaces |
| 6. Presentation | Slides, demo script, backup video, rehearsal timing |

## 11. Timeline (today 30 Oct → defence 09–14 Nov)

| Date | Milestone |
|------|-----------|
| 30–31 Oct | Spec approved, team roles fixed, Colab notebook skeleton, data split done |
| 1–3 Nov | E1–E4 trained, results table drafted |
| 4–5 Nov | E5 (optional), charts, best model chosen, `best.pt` saved |
| 5–6 Nov | Gradio app working with `best.pt`, tests passing |
| 7–8 Nov | Slides, backup demo video, two full rehearsals under 10 minutes |
| 9–14 Nov | Defence |

## 12. Risks

| Risk | Mitigation |
|------|------------|
| Colab GPU limit reached | Each run takes about 3–6 min on a T4 with frozen features. Save checkpoints to Google Drive after each run. |
| `best.pt` too large for GitHub (VGG16 with the original head is about 530 MB) | The E2–E4 heads keep it at about 160 MB. Store the weights on Google Drive or Hugging Face and download them in the README step. |
| `trash` class has low F1 | Report it honestly as a finding. Class-weighted loss is a possible "next step" slide. |
| Live demo fails | Pre-recorded video plus screenshots in the slides. |
