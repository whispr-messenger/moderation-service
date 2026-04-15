# Dataset -- Whispr Food Moderation

The repo has **four independent data acquisition pipelines** that feed three models. This document maps every source, every output location, every class taxonomy, and how they relate.

---

## 1. Quick map

```
                          ┌──────────────────────────────────────┐
                          │   HuggingFace (canonical, shared)    │
                          │  maia2000/food-classifier-dataset    │
                          │  maia2000/food-binary-dataset        │
                          └──────────────┬───────────────────────┘
                                         │
                ┌────────────────────────┼─────────────────────────┐
                │                        │                         │
   ┌────────────┴───────────┐ ┌──────────┴────────┐ ┌──────────────┴───────┐
   │ Pipeline A: scrape     │ │ Pipeline B:       │ │ Pipeline C: scrape   │
   │ Google/DuckDuckGo      │ │ scrape YouTube    │ │ DuckDuckGo (mirror   │
   │ images, 16 classes     │ │ + extract frames  │ │ of A, mobilenet dir) │
   │ → EfficientNet         │ │ → ViT-Video       │ │ → MobileNetV3        │
   └────────────────────────┘ └───────────────────┘ └──────────────────────┘

                          ┌──────────────────────────────────────┐
                          │ Pipeline D (fallback, in Colab):     │
                          │ Food-101 + Imagenette → 3-class      │
                          │ Runs when HF dataset returns < 1000  │
                          └──────────────────────────────────────┘
```

All four pipelines normalise to the same `frames/<class>/*.jpg` layout so any model can train on any source.

---

## 2. HF repos (canonical storage)

| Repo | Purpose | Used by |
|---|---|---|
| [`maia2000/food-classifier-dataset`](https://huggingface.co/datasets/maia2000/food-classifier-dataset) | Primary corpus. Either 16-class (legacy) or 3-class (`healthy`/`unhealthy`/`not_food`) depending on the latest push. | EfficientNet, MobileNet, ViT-Video |
| [`maia2000/food-binary-dataset`](https://huggingface.co/datasets/maia2000/food-binary-dataset) | Curated binary subset for the binary ViT release. | `vit_video_binary.ipynb` |

Cell 5b of every TF Colab notebook re-uploads `/content/frames/` to `food-classifier-dataset/frames/` after training, so subsequent runs hit the HF path in Cell 2 and skip scraping entirely.

---

## 3. Class taxonomies

The pipelines use **three different class layouts**. The trainer auto-detects which one is on disk and configures the head accordingly.

### 3a. Legacy 16-class

Used by Pipeline A/C (image scrapers) and the demo's `HEALTH_LABELS` rollup.

| Health group | Fine classes |
|---|---|
| **healthy** (8) | `fruits`, `vegetables`, `salads`, `seafood`, `grilled_meat`, `grain_bowls`, `soups`, `smoothies` |
| **unhealthy** (7) | `burgers`, `pizza`, `fried_food`, `desserts`, `candy_sweets`, `salty_snacks`, `sugary_drinks` |
| **not_food** (1) | `not_food` |

### 3b. 3-class (current TF training default)

Produced by Pipeline D (Food-101 + Imagenette fallback). Trains a softmax head over `{healthy, unhealthy, not_food}`.

### 3c. 2-class (binary)

Subset of 3b without `not_food`. Trains a sigmoid binary classifier. **Has no "not food" verdict** — every input is forced into one of two labels. Use only when upstream guarantees food images.

---

## 4. Pipeline A -- image scraping (EfficientNet)

**Code:** `src/efficientnet_lite_gpu/tools/fetch_google_dataset.py`
**Config:** `src/efficientnet_lite_gpu/tools/dataset_download_config.yaml` (16 classes, 20–40 search keywords each, target 600 images/class)
**Output:** `src/efficientnet_lite_gpu/train/dataset/Train/<class>/`

| | |
|---|---|
| **Sources** | DuckDuckGo (primary), Google Images, Bing Images (configurable) |
| **Format** | JPEG / PNG / WebP |
| **Classes** | 16 fine (taxonomy 3a) |
| **Per-class target** | 600 images |
| **Rate limits** | 0.5 s between downloads, 2 s between pages, 35 s between categories |
| **Quality filter** | Min file size 500 B, min resolution 200×200, content-type check |
| **Balancing** | Auto-trims excess to match the smallest class |

Browser-style headers reduce 403 errors from DuckDuckGo. To run:

```bash
src/efficientnet_lite_gpu/run_fetch_google_dataset.bat            # full
src/efficientnet_lite_gpu/run_fetch_google_dataset_dry_run.bat    # dry run
```

---

## 5. Pipeline B -- YouTube + frame extraction (ViT-Video)

**Code:** `src/vit_video/generatedata.py`, orchestrated by `src/vit_video/run_pipeline.py`
**Output:**
- Raw videos: `src/vit_video/food_data/raw_videos/<class>/*.mp4`
- Extracted frames: `src/vit_video/food_data/frames/<class>/*.jpg` (224×224)

| | |
|---|---|
| **Source** | YouTube via `yt-dlp` (`ytsearch<N>:<query>` syntax) |
| **Format** | MP4 (≤720p) → JPEG frames |
| **Classes** | 2 coarse (`healthy`, `unhealthy`) with **30 keywords each** |
| **Healthy keywords** | banana, apple, broccoli, salad, oatmeal, salmon, quinoa, yogurt, smoothie, avocado, steamed vegetables, stir fry, tofu, hummus, brown rice, berries, kale, sweet potato, lentil soup, edamame, grilled chicken, sushi, chickpea buddha bowl, overnight oats, roasted vegetables, poke bowl, fruit platter, … |
| **Unhealthy keywords** | cheeseburger, pizza, fries, donut, fried chicken, hot dog, pepperoni pizza, fast food breakfast sandwich, mozzarella sticks, corn dog, fish and chips, milkshake, ice cream sundae, chocolate cake, churros, cotton candy, funnel cake, cinnamon roll, brownie, cupcake, soda, energy drink, frappuccino, bubble tea, nachos, mukbang, … |
| **Videos / keyword** | Default 15 (CLI configurable) |
| **Frames / video** | Default 60, evenly spaced |
| **Frame size** | 224×224 |
| **Parallelism** | Up to 8 threads for frame extraction |
| **Rate limits** | None explicit; relies on `yt-dlp` internal throttling |

`run_pipeline.py` is idempotent: it skips YouTube if `raw_videos/` already exists, and skips frame extraction if `frames/` already exists. It also writes a video-level train/val/test split manifest so frames from the same source video stay in the same split (avoids leakage).

`yt-dlp` is forced to use the web client only (Android/iOS clients require PO tokens).

---

## 6. Pipeline C -- image scraping (MobileNetV3)

**Code:** `src/mobilenet_v3_small/tools/fetch_google_dataset.py` (identical to Pipeline A)
**Config:** `src/mobilenet_v3_small/tools/dataset_download_config.yaml`
**Output:** `src/mobilenet_v3_small/train/dataset/Train/<class>/`

Functionally identical to Pipeline A. Kept as a separate copy so the two model directories are self-contained and can scrape independently.

---

## 7. Pipeline D -- Food-101 + Imagenette fallback (Colab)

**Code:** `scripts/build_colab_notebooks.py` → emitted as Cell 2 of every TF Colab notebook.

Triggers when the HF dataset download returns fewer than 1000 frames. Builds a balanced 3-class corpus from public HF datasets — no API keys, no scraping, no rate-limit risk.

### 7a. Food-101 → healthy / unhealthy

Source: [`ethz/food101`](https://huggingface.co/datasets/ethz/food101) — ~75 000 train images, 101 fine food classes.

Cap: `PER_CLASS_CAP = 250` per source class → ~17 k food images.

The `FOOD101_TO_HEALTH` table covers ~70 unambiguous source classes; the remaining 31 are skipped.

| Source class | → Target |
|---|---|
| `caesar_salad`, `sashimi`, `grilled_salmon`, `bibimbap`, `pho`, `hummus`, `edamame`, `miso_soup`, `tofu`-style dishes | `healthy` |
| `hamburger`, `pizza`, `french_fries`, `donuts`, `cheesecake`, `ice_cream`, `nachos`, `chocolate_cake` | `unhealthy` |

**Mapping rationale:** salads, lean proteins, broth-based soups, whole grains → `healthy`. Burgers, pizza, fried, baked desserts, frozen sweets → `unhealthy`. Borderline classes (e.g. `fried_rice`, `shrimp_and_grits`, `pad_thai`) are mapped to whichever side dominates their nutrition profile; if you change them, retrain.

### 7b. Imagenette → not_food

Source: [`frgfm/imagenette`](https://huggingface.co/datasets/frgfm/imagenette), `320px` config — 10 non-food ImageNet classes (tench, English springer, cassette player, chainsaw, church, French horn, garbage truck, gas pump, golf ball, parachute).

Cap: `NOT_FOOD_CAP = 4000`. None overlap with food, so the entire pull is safe to label `not_food`.

### 7c. Output

```
healthy/      ~8 000 jpgs
unhealthy/    ~9 000 jpgs
not_food/     ~4 000 jpgs
```

~21 k images, ~40/45/19 split. Mild imbalance — the TF notebooks see it as-is (no `class_weight`); the ViT trainer computes inverse-frequency weights.

---

## 8. Common runtime layout (Colab)

```
/content/
├── frames/                       <-- raw images (any pipeline)
│   ├── healthy/
│   ├── unhealthy/
│   └── not_food/                 (only if 3-class)
└── binary/                       <-- symlink tree consumed by Keras
    ├── healthy/
    ├── unhealthy/
    └── not_food/
```

`frames/` holds actual JPEGs. `binary/` is rebuilt every run by symlinking `frames/<src>/img.jpg → binary/<target>/<src>_<orig>.jpg`. Empty class directories under `binary/` are deleted before training so Keras reports a clean class count.

For the ViT pipeline (Pipeline B) the structure under `food_data/` is the same idea — `raw_videos/` (sources) and `frames/` (training input).

---

## 9. Train / val split

All training pipelines use `validation_split=0.2` with a fixed `seed=42`. Reproducible across runs; checkpoints can resume mid-training without leaking val samples.

For Pipeline B (videos), the train/val/test split is taken at the **video** level by `run_pipeline.py`, so frames from the same source video can't appear in both train and val.

---

## 10. Known data quality notes

- **Label noise from Food-101 mapping** — borderline classes like `shrimp_and_grits`, `paella`, `risotto`. Expect a label-noise ceiling around 82–85 % val accuracy on the binary task; pushing beyond requires hand-relabelling, not more training. The ViT runs in this repo plateau at exactly this ceiling.
- **Imagenette is narrow** — 10 classes ≠ "everything that isn't food." If the deployed app sees food-adjacent non-food (cutlery, plates, tables), the `not_food` head may classify them as food. Add more not-food sources before relying on this verdict.
- **YouTube scraping yields temporally redundant frames** — 60 frames from one cooking video are highly correlated. Pipeline B mitigates by sampling evenly across the video and splitting at the video level, but you should still expect overfitting risk if `videos-per-keyword` is small.
- **Search-engine scraping yields contextually mislabelled images** — a "pizza" search returns pizza-themed t-shirts and emoji. The 200×200 / 500 B size filter helps but doesn't eliminate this. Spot-check before training.
- **EXIF orientation** — the demo app applies `PIL.ImageOps.exif_transpose` on upload; the training pipelines do not. Phone photos that aren't EXIF-corrected at training time may degrade ~1 % accuracy.
- **Truncated JPEGs** — the dataset contains a small number of partially-corrupt JPEGs (PIL prints `Truncated File Read`). They're tolerated, not pruned.

---

## 11. Reproducing the dataset locally

### Image scraping (Pipelines A / C)

```bash
cd src/efficientnet_lite_gpu
pip install -r requirements-fetch-only.txt
python tools/fetch_google_dataset.py --config tools/dataset_download_config.yaml
```

### Video scraping (Pipeline B)

```bash
cd src/vit_video
pip install -r requirements.txt
python run_pipeline.py --videos-per-keyword 15 --max-frames 60 --frame-size 224
```

### Public-dataset fallback (Pipeline D)

The mapping tables (`FOOD101_TO_HEALTH`, `NOT_FOOD_CAP`) live in `scripts/build_colab_notebooks.py` and are the source of truth. Copy them directly to keep parity with Colab.
