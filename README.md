# SmartBin

An image classifier that looks at a photo of one waste item and says which bin it belongs in. Built for ICT Assignment 2 (requirement option 1: an open-source dataset with a modified algorithm, comparing the results).

**Best model:** VGG16 transfer learning (experiment E1), **86.6 % test accuracy**, macro F1 0.851.

## Dataset

[TrashNet](https://github.com/garythung/trashnet) by Gary Thung and Mindy Yang: 2,527 photos of single items on a white background.

| cardboard | glass | metal | paper | plastic | trash |
|---|---|---|---|---|---|
| 403 | 501 | 410 | 594 | 482 | 137 |

The split is stratified 70 / 15 / 15 (1,768 train, 379 validation, 380 test) with seed 42, saved in `splits.csv` so every experiment uses the same images. We report macro F1 next to accuracy because the `trash` class is small.

## Experiments

Each experiment makes one main change from the run it builds on: E2 builds on E1, E3 on E2, E4 on E3, and E5 on E3. All ran for 10 epochs with batch size 32 and kept the weights with the best validation accuracy.

| ID | What changed | Trainable parameters | Test accuracy | Macro F1 | `trash` F1 | Training time |
|----|---|---:|---:|---:|---:|---:|
| **E1** | Baseline: frozen VGG16 features, last layer changed to 6 classes, SGD | 119,570,438 | **86.6 %** | **0.851** | 0.765 | 285 s |
| E2 | Smaller head: `Linear(25088→1024)`, ReLU, `Dropout(0.3)`, `Linear(1024→6)` | 25,697,286 | 84.7 % | 0.833 | 0.743 | 276 s |
| E3 | Optimiser: Adam, lr 1e-4, instead of SGD | 25,697,286 | 84.5 % | 0.833 | 0.757 | 280 s |
| E4 | Fine-tuning: last conv block unfrozen, lr 1e-5 for conv and 1e-4 for head | 32,776,710 | 84.2 % | 0.817 | 0.645 | 303 s |
| E5 | Different architecture: frozen MobileNetV2 | 7,686 | 75.5 % | 0.694 | 0.320 | 226 s |

Times are from an RTX 4060 laptop GPU.

![Experiment comparison](results/comparison.png)

### What we learned

- **Smaller head (E1→E2):** about 4.6× fewer trainable parameters shrink the saved model from 537 MB to 162 MB. Accuracy is 1.8 points lower, which is 7 of 380 test images and within noise. This isn't purely a size effect: E1 keeps VGG16's ImageNet-pretrained fully connected layers, while E2's new head starts from random weights and uses dropout 0.3 instead of 0.5. Training time barely changed (285 s vs 276 s).
- **Optimiser (E2→E3):** Adam and SGD gave practically the same result.
- **Fine-tuning (E3→E4):** no measurable gain on this seed. The VGG heads overfit quickly even without fine-tuning: training accuracy reached 99.1 % in E2 and 99.7 % in E3, while validation accuracy peaked at 82.8 % and 84.2 %. E4's validation accuracy moved between 81.8 % and 84.4 % and peaked at epoch 2 (tied with epoch 10, and the earlier epoch is kept), so the tested E4 model comes from epoch 2.
- **MobileNetV2 (E3→E5):** this is the smallest model, with a 9 MB file, and the least accurate. Training accuracy (81.6 %) and validation accuracy (77.0 %) were both still rising at epoch 10, so it was under-trained at this learning rate. More epochs may narrow the gap, but we didn't measure that. Its frozen BatchNorm layers must stay in eval mode; an earlier run that let their statistics change scored 73.2 % (see `results/training_log.txt`).

### Confusion matrix of the best model

![E1 confusion matrix](results/confusion_E1.png)

The most common mistakes are glass predicted as metal or plastic (7 each) and plastic predicted as metal (5). `trash` is the weakest class: 13 of 20 are correct, and 5 are predicted as paper.

## Limitations

- **Small differences between E1–E4 are not conclusive.** Each experiment ran once, with one seed. E1 and E4 differ by 9 of 380 test images, and the 95 % margin on 86.6 % is about ±3.4 points.
- **86.6 % is slightly optimistic.** The best model was chosen using test accuracy, so the test set is not fully independent of that choice. E1 ranks only third on validation accuracy (E4 84.4 %, E3 84.2 %, E1 83.6 %). Choosing by validation would pick E4, whose test accuracy of 84.2 % is just under our 85 % target.
- **Real photos may score lower.** TrashNet photos show one item on a white background. The model has not been measured on phone photos with cluttered backgrounds.
- **It can't recognise things that aren't waste.** It always picks one of the 6 classes. The "Not sure" warning appears when the top probability is below 50 %, but it doesn't catch everything. Without any warning, a photo of a building got "cardboard, 51 %" and a photo of a flower got "plastic, 68 %".
- **Bin names and colours are illustrative.** They are not an official Kazakhstan standard.

## Run the app

Python 3.10 or newer.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python app.py
```

Then open http://localhost:7860. On macOS or Linux, use `.venv/bin/` instead of `.venv/Scripts/`.

The app needs `weights/best.pt`, which isn't in the repository because it is 537 MB. Create it by training (below), or copy an existing `best.pt` into `weights/`.

## Train the models

The repository already includes our results, and experiments listed in `results/results.csv` are skipped. To retrain, delete that file first, then run:

```bash
.venv/Scripts/python run_experiments.py
```

This downloads TrashNet and trains every experiment, which takes about 25 minutes on an RTX 4060. If a run is interrupted, run the same command again: finished experiments are skipped.

`torch` from `requirements.txt` may be a CPU-only build, depending on your platform. To train on an NVIDIA GPU, install a CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/).

No GPU? Use [`notebooks/smartbin_training.ipynb`](notebooks/smartbin_training.ipynb) on Google Colab. It runs the same code. Our results came from the local GPU run, so the notebook hasn't been run on Colab itself yet.

## Tests

```bash
.venv/Scripts/python -m pytest
```

The tests run on CPU and don't download pretrained weights.

## Project layout

| File | Purpose |
|---|---|
| `model.py` | Classes, bin info, transforms, and the E1–E5 architectures |
| `data.py` | Dataset download, split, and loaders |
| `train.py` | Optimisers, training loop, and evaluation |
| `report.py` | Charts |
| `run_experiments.py` | Runs the whole pipeline |
| `app.py` | Gradio app |
| `notebooks/` | Colab version of the training run |
| `results/` | Results table, charts, per-class reports, and training logs (`training_log_part1.txt` has E1–E3, `training_log.txt` has E4 and the earlier E5 run, `training_log_E5_rerun.txt` has the final E5) |
| `Dockerfile` | Optional CPU container for the app (mount `weights/` at `/app/weights`). Not built or tested yet. |
