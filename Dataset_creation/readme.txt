Training Data :
MSCOCO
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
Waterbirds
https://huggingface.co/datasets/grodino/waterbirds



..........................Dataset Creation...........................




For waterbird
..................................................................


* For method one use: build_mist_waterbirds.py
* For method two use: build_probe_waterbirds.py
*For nethod three use: build_cord_waterbirds.py




For MSCOCO
..................................................................


* For method one use: build_method1_mist_coco.py
----------for inpaint use method1_MIST_inpaint.py
* For method two use: build_method2_probe_coco.py
*For nethod three use: build_method3_cord_coco.py



............................................................................Expected OUtput.....................................................

Expected output:
MIST:
output_root/
└── images/
    ├── train/
    │   ├── retain/<class>/*.png
    │   ├── forget/<class>/*.png
    │   └── forget_neg/<class>/*.png
    └── test/
        ├── retain/<class>/*.png
        ├── forget/<class>/*.png
        └── forget_neg/<class>/*.png
PROBE:
output_root/
└── images/
    ├── train/
    │   ├── retain/original_yes/<class>/*.png   # gt=1,pred=1
    │   ├── retain/original_no/<class>/*.png    # gt=0,pred=0
    │   ├── forget/original_yes/<class>/*.png   # gt=1,pred=0
    │   └── forget/original_no/<class>/*.png    # gt=0,pred=1
    └── test/
        ├── retain/original_yes/<class>/*.png
        ├── retain/original_no/<class>/*.png
        ├── forget/original_yes/<class>/*.png
        └── forget/original_no/<class>/*.png

CORD

output_root/
└── images/
    ├── train/
    │   ├── retain/<class>/*.png      # C ∩ S^c
    │   ├── forget/<class>/*.png      # C^c ∩ S
    │   ├── forget_neg/<class>/*.png  # C^c ∩ S^c
    │   └── both_pos/<class>/*.png    # C ∩ S
    └── test/
        ├── retain/<class>/*.png
        ├── forget/<class>/*.png
        ├── forget_neg/<class>/*.png
        └── both_pos/<class>/*.png
