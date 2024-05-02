# RWKV-LM-State-4bit

# WARNING: Eternal Debugging, pre-release.
This repo is forked from RWKV-LM

Test implement of RWKV v6 State-tuning with 4-bit quantization

if quant is disabled, it operates in bf16 training mode.


I have quantized the main weights of RWKV to 4 bits using Bitsandbytes and enabled State-Tuning. 

Additionally, I have configured the output checkpoint to only output time-state.

Quantizing to 4 bits can reduce VRAM usage by about 40%.

Ex. L32D2048 2B bf16 16GB to NF4 10GB

## This repo works
   - 1. Freeze Main Weight
   - 2. Quantize main weights 4bit via Bitsandbytes
   - 3. train
   - 4. output time_state only


## 4bit training
My training command is provided as follows:
```
python train.py --load_model "base_model/rwkv-16.pth"\
 --load_partial 1 \
 --wandb "RWKV-LM State-tuning" --proj_dir "1B6-State-Tuning"\
 --data_file "dataset/dataset" --train_type "states"\
 --data_type "binidx" --vocab_size 65536 --ctx_len 4096 \
 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 1 \
 --micro_bsz 1 --n_layer 24 --n_embd 2048 \
 --lr_init 1 --lr_final 0.01 \
 --warmup_steps 10 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
 --accelerator gpu --devices 1 --precision bf16 \
 --grad_cp 1 --my_testing "x060" \
 --strategy deepspeed_stage_1 \
 --quant 1 \
 --quant_type 'nf4'
```
## Merge to Base model
```
python merge_state.py <base_model.pth> <state_checkpoint.pth> <output.pth>
```


# And Thanks to:
   - RWKV-LM @BlinkDL
   - RWKV-PEFT @JL-er




# License
same with RWKV-LM

Apache 2.0


@ 2024 OpenMOSE
