# RAW2RGBNet
This is a PyTorch implement of RAW2RGBNet

## Training
```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py --name ednet_64_4_16_64 --model encoder_decoder --batchSize 32 --data_root /data1/kangfu/Datasets/RAW2RGB/ --checkpoint /data1/kangfu/Checkpoints/RAW2RGB/ --cuda --size 64
```

## Validation
```bash
python test-val.py --model mix3_deep_encoder_decoder --checkpoint /data1/kangfu/Checkpoints/RAW2RGB/mix3_deep_encoder_decoder_32_10_16_8_216_f_f_f/94.pth --output /data1/kangfu/Datasets/RAW2RGB/val_results --data /data1/kangfu/Datasets/RAW2RGB/RAW/

python psnr.py --data /data1/kangfu/Datasets/RAW2RGB/val_results --gt /data1/kangfu/Datasets/RAW2RGB/RGB/
```

## Testing
```bash
python test.py --model local_global_net --checkpoint /media/disk1/fordata/web_server/meikangfu/checkpoints_raw2rgb/17.pth --output /media/disk1/fordata/web_server/meikangfu/Datasets/RAW2RGB/submits --data /media/disk1/fordata/web_server/meikangfu/Datasets/RAW2RGB/Validation
```
