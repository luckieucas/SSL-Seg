## Semi-supervised Learning for Medical Image Segmentation (**SSL4MIS**)


# Usage


1. Train the 3d model
```
python train_semi-supervised.py --config train_config_3d.yaml
```
2. Train the 2d model
python train_semi-supervised.py --config train_config_3d.yaml
3. Test the model 
```
python test.py --model_path /path/to/trained_model
```
## Acknowledgement
* Part of the code is adapted from open-source codebase and original implementations of algorithms, we thank these author for their fantastic and efficient codebase, such as, [UA-MT](https://github.com/yulequan/UA-MT), [Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks) and [segmentatic_segmentation.pytorch](https://github.com/qubvel/segmentation_models.pytorch) . 
# SSL-Seg
