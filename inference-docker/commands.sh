#!/bin/bash
echo input folders:
ls  /input/images/kidney-transplant-biopsy-wsi-pas/
ls /input/images/tissue-mask/
echo model folder:
ls /opt/ml/model

echo running inference.py
python -u inference_multiclass_detection.py
echo inference.py finished