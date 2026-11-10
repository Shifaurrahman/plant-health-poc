# Tomato Pilot (PlantDoc)

This pilot adds tomato disease retrieval using PlantDoc field photos. It is separate from the brinjal pilot: own crops, own splits, own embedding cache, own accuracy table.

## 1. Scope

- Crop: tomato only (PlantDoc classes 19 to 27 in `data.yaml`)
- Class 27 (`Tomato_two_spotted_spider_mites_leaf`) is excluded: it has 2 test crops and 0 index crops, so it cannot be matched.
- Healthy class (25, `Tomato_leaf`) is a healthy state, not a disease candidate.
- Brinjal is not changed. Its cache and splits are untouched.

## 2. Data

Source: PlantDoc, Roboflow YOLO export (CC BY 4.0). Citation:

```
Singh et al. 2019, "PlantDoc: A Dataset for Visual Plant Disease Detection", arXiv:1911.10317
```

Folder kept outside the project: `C:\Users\USER\Downloads\archive` (`train/`, `valid/`, `test/` with `images/` and `labels/`, plus `data.yaml`).

## 3. Pipeline

| Step | Script / file | Output |
|---|---|---|
| 1. Crop tomato boxes | `scripts/plantdoc_crop_tomato.py` | `data/raw/plantdoc_tomato/<class>/<split>_<n>.jpg`, `metadata.csv` |
| 2. Grouped split | `scripts/plantdoc_tomato_split.py` | `data/splits/plantdoc_tomato_index.csv`, `plantdoc_tomato_test.csv` |
| 3. Drop spider-mite class | `scripts/plantdoc_drop_class.py` | Same two CSVs, class 27 removed |
| 4. Leak checks | inline checks (below) | Must print 0 |
| 5. Evaluation / cache | `scripts/evaluate_retrieval.py` | `data/embeddings/plantdoc_tomato_gemma2.npz`, results JSON |

### Commands (run from `backend\`)

```cmd
python scripts/plantdoc_crop_tomato.py "C:\Users\USER\Downloads\archive" data\raw\plantdoc_tomato
python scripts/plantdoc_tomato_split.py data\raw\plantdoc_tomato\metadata.csv data\splits
python scripts/plantdoc_drop_class.py
```

Expected counts after step 3: index 1,270 crops, test 345 crops.

### Leak checks

Photo-level duplicates across PlantDoc splits (base name, augmentation suffix removed) must be 0:

```cmd
python -c "import csv,re; from collections import defaultdict; rows=list(csv.DictReader(open('data/raw/plantdoc_tomato/metadata.csv', encoding='utf-8'))); by=defaultdict(set); [by[re.split(r'\.rf\.', r['source_image'].split('__',1)[1])[0]].add(r['split_origin']) for r in rows]; d={s:v for s,v in by.items() if len(v)>1}; print('base photos in more than one split:', len(d))"
```

Missing files must be 0:

```cmd
python -c "import csv,os; r=list(csv.DictReader(open('data/splits/plantdoc_tomato_index.csv',encoding='utf-8'))); miss=[x['relative_path'] for x in r if not os.path.exists(os.path.join('data/raw',x['relative_path']))]; print('missing files:',len(miss))"
```

## 4. Evaluation

Use a tomato-specific cache name so brinjal's cache is not overwritten:

```cmd
set PH_EMBED_CACHE_NAME=plantdoc_tomato_gemma2.npz
python scripts/evaluate_retrieval.py --dataset data/raw --splits data/splits --index-file data/splits/plantdoc_tomato_index.csv --test-file data/splits/plantdoc_tomato_test.csv --batch 8
```

Note: `--dataset data/raw` is required. The CSV paths already start with `plantdoc_tomato/`.

Report accuracy per class. Do not mix with brinjal numbers.

## 5. Cache design

One cache per crop, same embedding model (EmbeddingGemma 2):

```
data/embeddings/
  embeddinggemma2.npz          brinjal
  plantdoc_tomato_gemma2.npz   tomato
```

Reason: a tomato vector must never win a brinjal query. The app must choose the cache by the case's crop. Current `image_retrieval` reads one `EMBED_CACHE`, so it needs a crop-based file choice before tomato goes live.

## 6. Records and app enablement (not yet done)

Tomato is not supported until all of these are true:

1. Expert-approved records exist with `crop: tomato` for classes 19 to 26 (class 25 healthy).
2. Cache file chosen by crop (code change above).
3. Accuracy run done and reviewed.
4. `PH_SUPPORTED_CROPS=brinjal,tomato` set.

Until then, a tomato photo must get `unsupported_crop` or "no approved match", never a brinjal answer.

## 7. Running on Colab (for manager)

Full setup on Colab, same order as the brinjal notebook:

1. Runtime → Change runtime type → T4 GPU.
2. Mount Drive. Keep `plant-health/` on Drive with `backend/`, `data/splits/`, `data/raw/plantdoc_tomato/` and `Qwen3-VL-Embedding/` (if used).
3. Install: `pip install -q -U sentence-transformers transformers pillow soundfile torchcodec`.
4. Set `os.chdir` to `.../plant-health/backend`.
5. Run the evaluation command above (`!python scripts/evaluate_retrieval.py ...`).
6. Download `plantdoc_tomato_gemma2.npz` and the results JSON back to the PC.

Set `PH_EMBED_CACHE_NAME` in the same cell before the run, since each `!` command is a new process.

## 8. Known limits

- PlantDoc is field photos from the web, not the same camera or farm as SLIF-Brinjal. Expect different accuracy.
- Mosaic virus and similar classes are weak in every dataset so far.
- Class 27 excluded; no spider-mite results.
- Dataset images are resized by Roboflow in some exports; this export is the original-size YOLO set.

## 9. Status

- [x] Crop step (1,617 crops)
- [x] Grouped split (index 1,270 / test 347, then 345 after class 27 removed)
- [x] Leak checks: 0 and 0
- [ ] Evaluation run
- [ ] Records for tomato
- [ ] Crop-based cache selection in backend
- [ ] App enablement