## 사용 방법
```bash
cd copy_paste
```

```
# 이미지 별 mask png 파일 만들기
python get_coco_mask.py --input_dir {데이터셋 경로} --output_dir {Mask png 파일 만들어질 경로} --split train_all
```

```
# copy paste
python copy_paste.py --input_dir {.jpg .png 파일 있는 경로} --output_dir {copy_paste 결과물 저장 경로} --lsj {Large Scale Jitter 사용 여부}
```

후에 ipynb 파일 사용하여 coco format json 파일 생성