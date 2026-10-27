# Dataset sources for the Plant Health POC

Check the licence on each page before use. Licences were not verified.

## Sri Lankan datasets (use these first)

### 1. SLIF-Brinjal (Sri Lanka, in-field, brinjal)
- Page: https://data.mendeley.com/datasets/6yg6vktrc2/2 (version 2, DOI 10.17632/6yg6vktrc2.2)
- Files on the page:
  - `SLIF-Brinjal.zip`, 2.35 GB: the images. Download this one.
  - `metadata.csv`, 2.33 MB: labels and metadata. Download this too.
- Inside the zip: Raw Dataset (1,459), Phase I (4,377), Phase II (8,987). Use the class folders.
- Classes: Bacterial Blight, Bacterial Leaf Spot, Bacterial Wilt, Cercospora Leaf Spot, Little Leaf, Mosaic Virus, Powdery Mildew, Healthy.
- Cite the paper shown on the page (Scientific Data, under review).
- Folder: `data/raw/sri_lanka_brinjal/`

### 2. Sri Lankan Tea Leaf Dataset
- Version 3 (`/datasets/sjmy6k24d6/3`) has been **removed by the author**. Do not use it.
- Use version 5 (16 Sept 2026): https://data.mendeley.com/datasets/sjmy6k24d6/5
- Version 6 is under embargo.
- The earlier version had 1,791 raw images (6 classes: Algal leaf spot, Black blight, Blister blight, Gray blight, Spider mites, Healthy). **Open version 5 and check its class list and counts**, because the content may differ from version 3.
- Folder: `data/raw/sri_lanka_tea/`

## Kaggle datasets

| Dataset | Kaggle slug | Size | Classes | Caveat |
|---|---|---|---|---|
| New Plant Diseases Dataset | vipoooool/new-plant-diseases-dataset | about 3 GB, about 87,900 files | 38 (PlantVillage crops) | Plain backgrounds, so weak for field photos. Use for training only, not for the field test set. Folder: `data/raw/kaggle_new_plant_diseases/` |
| Rice Leaf Disease Classification | anshulm257/rice-disease-dataset | 3,829 images | Bacterial Leaf Blight, Brown Spot, Healthy, Leaf Blast, Leaf Scald, Sheath Blight | Use only if rice becomes a priority. No Sri Lankan rice dataset found. Folder: `data/raw/kaggle_rice/` |

## Not used
- Mendeley tea sets from India or Bangladesh (for example 94fzcdz8gz, mz598gxfw9, rrff4vb35m, 744vznw5k2): not Sri Lankan. Do not label them as Sri Lankan.
- Mendeley "Plant Leaf Disease Recognition" (5g238dv4ht): Bangladeshi, covers gourd, hibiscus, papaya, zucchini.
- MedLEA medicinal leaf images: species identification, not disease.

## Finding more Kaggle datasets from the terminal
```bash
kaggle datasets list -s "plant disease leaf" --sort-by votes
kaggle datasets list -s "soybean leaf disease"
kaggle datasets list -s "corn maize leaf disease"
```
The first column (`ref`) is the slug to use with `kaggle datasets download -d <ref>`.
Or search in the browser: https://www.kaggle.com/datasets?search=plant+disease

## Important caveat
The Kaggle New Plant Diseases images use plain backgrounds, unlike farmer photos. Use the Sri Lankan in-field data as the main evaluation set.