# Once Quantized for All

## Citation 
If you find our code useful for your research, please consider citing:

@article{shen2020once,
  title={Once Quantized for All: Progressively Searching for Quantized Efficient Models},
  author={Shen, Mingzhu and Liang, Feng and Li, Chuming and Lin, Chen and Sun, Ming and Yan, Junjie and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2010.04354},
  year={2020}
}

## Statement

This repo contains the evaluation code of 4/3/2 bit OQA checkpoints.
The entire code(including training) will be released when it's properly arranged.

This repo is based on the [Once for All](https://github.com/mit-han-lab/once-for-all).

## Build env

Following the setup requirements in [Once for All](https://github.com/mit-han-lab/once-for-all).

## Eval

checkpoints can be found on anonymous [google drive](https://drive.google.com/drive/folders/1yGymACKYzCVns3KPs5hbuT25jlpZIrWh?usp=sharing)

### Evaluate random subnets

```console
# Only support single GPU
python -u eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b4 
```
### Evaluate pareto subnets

####Pareto 4bit
```console
# FLOPs 55.385 Params 4.230 OQA_top1 60.592
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b4 -is 128 -ks 3,3,5,3,3,5,5,7,3,5,5,3,5,5,3,5,3,5,7,5 -es 3,3,4,4,3,4,3,3,3,3,3,4,4,3,6,6,6,4,3,3 -d 2,2,3,2,3 

# FLOPs 68.360 Params 3.523 OQA_top1 63.928
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b4 -is 152 -ks 5,5,5,5,5,3,3,7,5,7,7,5,5,7,5,7,5,3,3,5 -es 3,3,4,6,3,3,3,3,3,3,4,4,3,3,6,6,4,3,3,6 -d 2,2,2,2,2 

# FLOPs 82.601 Params 4.482 OQA_top1 65.962
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b4 -is 156 -ks 3,5,5,3,5,3,5,3,3,3,7,7,3,5,7,7,7,3,3,7 -es 3,4,6,6,3,4,4,3,3,3,4,3,4,3,3,4,4,4,6,3 -d 2,2,2,2,3 

# FLOPs 120.970 Params 4.084 OQA_top1 68.948
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b4 -is 188 -ks 7,3,7,5,5,3,5,5,7,5,5,5,5,5,7,7,7,7,3,3 -es 4,3,3,4,3,3,3,6,4,3,3,6,4,4,6,4,4,4,3,3 -d 2,2,2,2,3 

# FLOPs 168.057 Params 5.440 OQA_top1 71.264
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b4 -is 184 -ks 5,3,7,3,7,3,5,7,7,3,7,3,3,3,5,5,5,5,3,7 -es 4,3,6,3,4,4,3,6,6,3,4,3,4,4,4,3,6,6,6,6 -d 2,3,3,3,3 

# FLOPs 222.192 Params 5.331 OQA_top1 72.178
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b4 -is 224 -ks 5,3,3,5,3,7,5,3,3,5,3,5,5,3,7,7,7,7,7,3 -es 4,3,4,3,3,4,6,3,4,4,4,3,4,4,6,4,4,4,6,3 -d 2,2,2,3,4 

# FLOPs 286.910 Params 6.338 OQA_top1 73.504
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b4 -is 216 -ks 5,3,5,3,5,5,7,3,3,7,7,5,7,7,3,3,5,5,3,7 -es 4,4,3,4,3,3,6,4,3,4,4,4,6,6,4,3,6,4,6,6 -d 2,4,2,4,4 
```

####Pareto 3bit
```console
# FLOPs 59.499 Params 3.911 OQA_3b_top1 53.416
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b3 -is 128 -ks 7,3,5,3,5,3,3,3,5,5,3,3,5,5,5,5,3,7,7,7 -es 3,4,3,4,3,3,6,6,4,4,6,6,6,4,4,4,3,4,3,6 -d 2,2,3,2,2 -bit 3

# FLOPs 73.368 Params 3.650 OQA_3b_top1 59.040
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b3 -is 152 -ks 3,3,5,3,3,3,5,5,7,3,5,3,3,7,7,5,3,3,7,3 -es 3,4,3,4,4,4,3,6,3,3,6,6,6,3,3,6,4,3,4,3 -d 2,2,2,2,2 -bit 3

# FLOPs 119.535 Params 4.281 OQA_3b_top1 64.088
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b3 -is 172 -ks 5,3,3,5,3,7,7,3,5,7,5,5,5,5,5,3,7,7,3,5 -es 3,3,3,3,3,4,3,3,4,4,6,6,6,3,4,4,6,4,3,4 -d 2,2,2,3,2 -bit 3

# FLOPs 174.790 Params 5.015 OQA_3b_top1 67.016 OQA_2b_top1 57.558
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b3 -is 192 -ks 3,3,7,5,7,5,3,3,3,3,3,3,3,5,5,5,7,7,7,5 -es 3,3,3,6,4,6,3,6,4,4,3,4,6,4,4,3,3,6,4,3 -d 2,4,3,3,3 -bit 3

# FLOPs 219.644 Params 5.762 OQA_3b_top1 69.046 OQA_2b_top1 60.236
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b3 -is 208 -ks 5,5,5,5,5,3,5,5,5,3,7,3,5,5,3,7,3,7,5,3 -es 3,3,3,3,6,4,4,4,6,4,4,4,6,3,3,4,6,6,6,3 -d 2,2,4,3,4 -bit 3

# FLOPs 292.019 Params 6.280 OQA_3b_top1 70.450 OQA_2b_top1 61.324
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b3 -is 212 -ks 7,5,5,3,3,5,5,3,3,5,7,7,7,5,7,5,7,3,3,3 -es 3,4,4,3,6,4,4,3,6,4,3,4,3,4,4,6,6,6,6,4 -d 2,4,3,4,4 -bit 3
```

####Pareto 2bit
```console
# FLOPs 59.499 Params 3.911 OQA_2b_top1 42.390
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b2 -is 128 -ks 7,3,5,3,5,3,3,3,5,5,3,3,5,5,5,5,3,7,7,7 -es 3,4,3,4,3,3,6,6,4,4,6,6,6,4,4,4,3,4,3,6 -d 2,2,3,2,2 -bit 2

# FLOPs 87.848 Params 3.694 OQA_2b_top1 50.428
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b2 -is 160 -ks 3,5,3,7,5,3,3,3,5,3,5,5,3,5,3,5,3,3,7,7 -es 4,3,4,4,6,6,3,3,4,4,4,6,6,4,6,4,3,3,4,4 -d 2,2,2,2,2 -bit 2

# FLOPs 135.267 Params 4.078 OQA_2b_top1 55.708
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b2 -is 188 -ks 7,5,5,5,7,3,5,5,3,3,3,3,3,7,3,5,7,7,5,3 -es 4,3,3,4,4,6,4,4,6,3,3,4,6,3,6,6,6,4,4,6 -d 2,2,2,2,2 -bit 2 

# FLOPs 199.504 Params 5.066 OQA_2b_top1 59.780
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b2 -is 208 -ks 5,5,5,3,5,5,5,5,3,3,7,3,3,3,7,5,5,5,3,7 -es 3,3,6,4,6,6,3,3,4,3,3,3,6,3,4,4,6,6,4,4 -d 2,3,2,3,3 -bit 2 

# FLOPs 289.256 Params 5.321 OQA_2b_top1 62.008
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b2 -is 220 -ks 5,5,5,3,3,7,3,5,5,3,3,5,5,7,5,7,7,5,3,3 -es 3,3,3,4,6,6,4,6,6,3,4,6,4,4,3,3,6,4,6,4 -d 3,3,4,4,3 -bit 2 

# FLOPs 455.741 Params 7.406 OQA_2b_top1 64.148
python eval_oqa_net.py --net oqa_mbv3_d234_e346_k357_w1.0_b2 -is 224 -ks 5,5,7,5,7,5,5,3,5,5,3,5,3,7,5,5,7,7,3,7 -es 6,6,4,4,6,4,6,4,6,4,6,6,6,6,6,6,6,6,6,6 -d 4,4,4,4,4 -bit 2 
```
