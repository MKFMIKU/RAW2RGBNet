# RAW2RGBNet
This is a PyTorch implement of RAW2RGBNet

## Training
```bash
 CUDA_VISIBLE_DEVICES=1,2,3 python train.py --name mix4_deep_encoder_decoder_32_8_10_8_144_f --model mix4_deep_encoder_decoder --batchSize=18 --data_root ~/ram_data/RAW2RGB/ --checkpoint /data1/kangfu/Checkpoints/RAW2RGB/ --cuda --size 192 --lr 1e-5 --n-epoch=100 --resume /data1/kangfu/Checkpoints/RAW2RGB/mix4_deep_encoder_decoder_32_8_10_8_144_f/16.pth --start-epoch=17
```

## Testing
```bash
python test.py --model local_global_net --checkpoint /media/disk1/fordata/web_server/meikangfu/checkpoints_raw2rgb/17.pth --output /media/disk1/fordata/web_server/meikangfu/Datasets/RAW2RGB/submits --data /media/disk1/fordata/web_server/meikangfu/Datasets/RAW2RGB/Validation
```
