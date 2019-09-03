# RAW2RGBNet
This is a PyTorch implement of RAW2RGBNet (haoyu)

## Training
```bash
 CUDA_VISIBLE_DEVICES=1,2,3 python train.py --name mix4_deep_encoder_decoder_32_8_10_8_144_f --model mix4_deep_encoder_decoder --batchSize=18 --data_root ~/ram_data/RAW2RGB/ --checkpoint /data1/kangfu/Checkpoints/RAW2RGB/ --cuda --size 192 --lr 1e-5 --n-epoch=100 --resume /data1/kangfu/Checkpoints/RAW2RGB/mix4_deep_encoder_decoder_32_8_10_8_144_f/16.pth --start-epoch=17
```

## Testing
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model mix3_deep_encoder_decoder --checkpoint ./80.pth --output /data1/kangfu/Datasets/RAW2RGB/val_results --data ~/ram_data/RAW2RGB/Validation
```

## Full Validation
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model mix3_deep_encoder_decoder --checkpoint ~/Codes/RAW2RGB/80.pth --output ~/haoyu/fullres_val_results --data ~/ram_data/RAW2RGB/FullResValidation
```