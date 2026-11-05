
Train and WAG and Acccuracy calculation 
---------------------------------------------------





CLIP PURGE D
------------------------------


python -u "clip_D.py" \
  --waterbirds-root "/datasets/waterbirds/waterbird_complete95_forest2water2" \
  --metadata-csv "/waterbirds/waterbird_complete95_forest2water2/metadata.csv" \
  --purge-d-root "/datasets/Waterbirds_PURGE_D" \
  --clip-checkpoint "/models/RN50_openai.pt" \
  --output-dir "/outputs/Waterbirds_PURGE_D_WGA_final_v2" \
  --device cuda: \
  --epochs 5 \
  --batch-size 64 \
  --num-workers 0 \
  --baseline-lr 1e-4 \
  --purge-lr 1e-5 \
  --samples-per-class 100 \
  --lambda-forget 1.0 \
  --beta 0.1 \
  --group-loss-weight 2.0 \
  --accuracy-tolerance 0.02 \
  --seed 42
  
  
  
CLIP PURGE H
-------------------------------------------------------


python -u "clip_H.py" \
  --waterbirds-root "/datasets/waterbirds/waterbird_complete95_forest2water2" \
  --metadata-csv "/datasets/waterbirds/waterbird_complete95_forest2water2/metadata.csv" \
  --purge-h-root "/datasets/Waterbirds_PURGE_H" \
  --clip-checkpoint "/models/RN50_openai.pt" \
  --output-dir "/outputs/Waterbirds_PURGE_H_pretrained_fullgroups" \
  --device cuda: \
  --epochs 5 \
  --batch-size 64 \
  --num-workers 0 \
  --purge-lr 1e-5 \
  --samples-per-class 100 \
  --lambda-forget 1.0 \
  --beta 0.1 \
  --group-loss-weight 5.0 \
  --accuracy-tolerance 0.03 \
  --seed 42
  

CLIP PURGE M
----------------------------------------------

 python -u "/clip_M.py" \
  --purge-m-root "/datasets/Waterbirds_PURGE_M" \
  --clip-checkpoint "/home/hq1351@ad.wayne.edu/home/hal/models/RN50_openai.pt" \
  --baseline-checkpoint "/outputs/Waterbirds_PURGE_M_balancedNPO_v1/original/best_finetune.pt" \
  --output-dir "/outputs/Waterbirds_PURGE_M_accuracy_priority_v1" \
  --device cuda: \
  --epochs 5 \
  --batch-size 64 \
  --num-workers 0 \
  --purge-lr 1e-5 \
  --samples-per-class 100 \
  --lambda-forget 1.0 \
  --beta 0.1 \
  --accuracy-tolerance 0.03 \
  --min-val-wga 0.80 \
  --seed 42
