# RAW2RGBNet
This is a PyTorch implement of RAW2RGBNet. Our Team: The First Team of Hogwarts School got 22.34dB in the validation set.
For more details please refer to the official website of the challenge: https://competitions.codalab.org/competitions/20158#results

## Training
```bash
python train.py --name full_mix3_deep_encoder_decoder --model full_mix3_deep_encoder_decoder --batchSize 16 --data_root /data1/kangfu/Datasets/RAW2RGB/ --checkpoint /data1/kangfu/Checkpoints/RAW2RGB/ --cuda --size 64
```

## Validation
```bash
python test-val.py --model mix3_deep_encoder_decoder --checkpoint /data1/kangfu/Checkpoints/RAW2RGB/mix3_deep_encoder_decoder_32_10_16_8_216_f_f_f/94.pth --output /data1/kangfu/Datasets/RAW2RGB/val_results --data /data1/kangfu/Datasets/RAW2RGB/RAW/

python psnr.py --data /data1/kangfu/Datasets/RAW2RGB/val_results --gt /data1/kangfu/Datasets/RAW2RGB/RGB/
```

## Testing
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model mix3_deep_encoder_decoder --checkpoint ./80.pth --output /data1/kangfu/Datasets/RAW2RGB/val_results --data ~/ram_data/RAW2RGB/Validation
```

## Testing Full-Resolution Images on Single Titan XP (12GB)
```bash
CUDA_VISIBLE_DEVICES=0 python test-pad.py --model full_mix3_deep_encoder_decoder --checkpoint ./112.pth --output /data1/kangfu/Datasets/RAW2RGB/testing_full_results_full_mix3_bacth_224_ep_112  --data /data1/kangfu/Datasets/RAW2RGB/FullResTestingPhoneRaw
```

## Testing Full-Resolution Images on Single Tesla M40 (24GB)
```bash
python3 test-full.py --model full_mix3_deep_encoder_decoder --checkpoint ./114.pth --output ../testing_full_results_full_mix3_bacth_224_ep_114  --data ../FullResTestingPhoneRaw/
```

## Reproduce results in the challenge submission
You can download the pre-trained model from here [114.pth](https://cuhko365-my.sharepoint.com/:u:/g/personal/219019003_link_cuhk_edu_cn/EZrS367uMMlPjVEQ41j1N30B-4d6fcfNESWNi0JPH2Pyfg?e=IlRYiU) [115.pth](https://cuhko365-my.sharepoint.com/:u:/g/personal/219019003_link_cuhk_edu_cn/Ea7hSVs-cXFHhKGxiTAt6BUBCh66brqiaeiqSNRfigoc2Q?e=wD8WwN)
```bash
# For track 1
# generate results using the 114.pth and the 115.pth respectively
python test.py --model full_mix3_deep_encoder_decoder --checkpoint ./114.pth --output /data1/kangfu/Datasets/RAW2RGB/validation_results_full_mix3_bacth_224_ep_114 --data /data1/kangfu/Datasets/RAW2RGB/Validation

python test.py --model full_mix3_deep_encoder_decoder --checkpoint ./115.pth --output /data1/kangfu/Datasets/RAW2RGB/validation_results_full_mix3_bacth_224_ep_115--data /data1/kangfu/Datasets/RAW2RGB/Validation

# ensemble the results from 104.pth and 105.pth
python result_ensemble.py --data /data1/kangfu/Datasets/RAW2RGB/testing_results_full_mix3_bacth_224_ep_114,/data1/kangfu/Datasets/RAW2RGB/testing_results_full_mix3_bacth_224_ep_115 --output /data1/kangfu/Datasets/RAW2RGB/testing_results_ensemble_114_115

# For track 2
# generate results using the 115.pth only
python3 test-full.py --model full_mix3_deep_encoder_decoder --checkpoint ./115.pth --output ../testing_full_results_full_mix3_bacth_224_ep_114  --data ../FullResTestingPhoneRaw/

```


## Contact
If you have any questions about the code, please contact kangfumei@link.cuhk.edu.cn
