Evaluation Tools for CityPersons
===

### Ground-Truth Preparing
1. If you want to evaluate $MR^{-2}$ for body/face, use instances_val_bf_new.json in [ground-truth](https://drive.google.com/drive/folders/12ypJ8gB7v4T1_blYraGslCRK9hXuNGBP).
2. If you want to evaluate $mMR^{-2}$, use instances_val_bhfmatch_new.json in [ground-truth](https://drive.google.com/drive/folders/12ypJ8gB7v4T1_blYraGslCRK9hXuNGBP).

### Results Preparing
1. If you want to evaluate $MR^{-2}$ for body/face, dump your results to a standard COCO-format JSON file in which the "body" and "face" are two categories in parallel. You can refer to bbox.json in [demo-results](https://drive.google.com/drive/folders/1ojzw7k729qQ-qf1N_Ngc5T91iAPS75i5) for more details.
2. If you want to evaluate $mMR^{-2}$, dump your results following format below:
```json
[
...
{"image_id": 497, 
 "category_id": 1, 
 "bbox": [1030.8, 425.1, 15.2, 37.0], 
 "f_bbox": [1120.5, 489.1, 10.2, 12.0], 
"score": 0.32, "f_score": 0.28},
...
]
```
If a body does not has a face matched, use [0.0, 0.0, 1.0, 1.0] and 0.0 as placeholders for "f_bbox" and "f_score" respectively.
You can refer to bf_match_bbox.json in [demo-results](https://drive.google.com/drive/folders/1ojzw7k729qQ-qf1N_Ngc5T91iAPS75i5) for more details.

### Evaluation

* Evaluate $mMR^{-2}$

```shell
python main.py -dt bf_match_bbox.json -gt instances_val_bhfmatch_new.json -t mmr
```

* Evaluate $MR^{-2}$ for body or face

```shell
python main.py -dt bbox.json -gt instances_val_bf_new.json -t mr_body
```

```shell
python main.py -dt bbox.json -gt instances_val_bf_new.json -t mr_face
```