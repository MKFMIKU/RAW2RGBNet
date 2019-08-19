# RAW2RGBNet
This is a PyTorch implement of RAW2RGBNet

## Training
```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py --name ednet_64_4_16_64 --model encoder_decoder --batchSize 32 --data_root /data1/kangfu/Datasets/RAW2RGB/ --checkpoint /data1/kangfu/Checkpoints/RAW2RGB/ --cuda --size 64
```

## Testing
```bash
python test.py --model local_global_net --checkpoint /media/disk1/fordata/web_server/meikangfu/checkpoints_raw2rgb/17.pth --output /media/disk1/fordata/web_server/meikangfu/Datasets/RAW2RGB/submits --data /media/disk1/fordata/web_server/meikangfu/Datasets/RAW2RGB/Validation
```
