# HyP2-Loss
Official PyTorch implementation of paper HyP$^2$ Loss:Beyond Hypersphere Metric Space for Multi-label Image Retrieval
### Requirements
```
NVIDIA GPU + CUDA (may CuDNN) and corresponding PyTorch framework
Python 3
```
### Datasets
##### Flickr-25K/NUS-WIDE
We recommend you to follow [] to prepare Flickr-25K and NUS-WIDE images.
##### VOC2007/VOC2012
Please run the training command with `--dataset voc2007/voc2012` directly and the voc2007/voc2012 dataset will be downloaded automatically.
### Training
```
python retrieval --dataset [dataset] --backbone [backbone] --hash_bit [hash_bit]
```
Arguments (default value)
```
--dataset:        dataset                                         [(voc2007), voc2012, flickr, nuswide]
--backbone:       backbone network for feature extracting         [(googlenet), alexnet, resnet]
--hash_bit:       length of hash bits                             [(48), 12, 16, 24, 32, 36, 64]
```
Other optional arguments
```
--beta:           a hyper-parameter to balance the contribution of proxy loss and pair loss      
--batch_size      the size of a mini-batch                                                            
```
### Inference
Add `--test` after the training command. Make sure there is a corresponding `.ckpt` file in the `./result/` directory.
```
python retrieval --dataset [dataset] --backbone [backbone] --hash_bit [hash_bit] --test
```
### Performance

### Citation
If you use this method or this code in your research, please cite as:
```

```
