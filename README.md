# PIXEL_ATTACK
This repository explores adversarial attacks that challenge the robustness of image classification models. Our goal is to design and evaluate algorithms capable of identifying minimal pixel perturbations that successfully mislead deep learning classifiers.

---

## 📄 Main Files

### 📊 **Metric Scripts — `PIXEL-ATTACKS/METRICS FILES/`**

#### `results_analysis.py`

Extracts and analyses key metrics such as:

* **Covered pixels**
* **Adversarial images found**
* **Evaluations until an adversarial pixel is discovered**
* **Success rate**
* **Distortion**
* **Execution time**

  * Uses the `time_file` in append mode (`'a'`)
  * Currently set to retrieve only the last `nruns`

**Outputs produced:**

* `results/results_full.txt` — full information: all runs, means, and standard deviations
* `results/results_simp.txt` — simplified version: means and standard deviations per approach

---

### 🖼️ **Image & Graph Generation — `make_images.py`**

Runs multiple approaches and models, producing graphs over evolutionary generations.

**Outputs stored in:** `results/graphics/`

**Generated charts:**

* `{modelName}_acc.png` — accumulated adversarial images
* `{modelName}_adv.png` — adversarial images per generation
* `{modelName}_fit_avg.png` — average fitness per generation
* `{modelName}_fit_best.png` — best fitness per generation
* `{modelName}_fit.png` — combined fitness graph

---

## 🔎 Distortion Measurement

Distortion quantifies the difference between an adversarial image and its original version.

Since only one pixel is changed, it is computed as:

```
dist = |IR − PR| + |IG − PG| + |IB − PB|
```

Where `IR, IG, IB` are the original pixel values and `PR, PG, PB` are the perturbed values.

### Scripts involved:

#### `CountDif_orig_img.py`

* Calculates the distortion of each successful adversarial pixel
* Inputs:

  * `./results/{modelName}/metrics_mean_img.csv` — image index
  * `./results/{modelName}/{approach}/run_{i_run}/img_{i_img}/success_file.csv` — genotype of successful pixels

#### `StatsDif.py`

* Computes the average distortion per run
* Input: `./results/{modelName}/`
* Output: `./results/{modelName}/run/img/new_success_file.csv`

#### `medias_dif_per_model.py`

* Computes the average distortion per model
* Input: `./results/{modelName}/run/img/new_success_file.csv`
* Output: `./results/{modelName}/difMeans.csv`

---

## 🔍 Local Search — `local_search.py`

After adversarial pixels are initially detected, this script searches around those pixels to find alternatives that may be:

* more effective
* less perceptible

---

# RUNNING CODE:

Adapt as you need:
\>\> TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run_cycle_ga.py 