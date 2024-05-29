python -m training.train --config configs/train_single_model_lora.yaml --model_type dino_vitb16 --feat_type 'cls' --stride '16' &
CUDA_VISIBLE_DEVICES=1 python -m training.train --config configs/train_single_model_lora.yaml --model_type clip_vitb32 --feat_type 'embedding' --stride '32' &
CUDA_VISIBLE_DEVICES=2 python -m training.train --config configs/train_single_model_lora.yaml --model_type open_clip_vitb32 --feat_type 'embedding' --stride '32' &
CUDA_VISIBLE_DEVICES=3 python -m training.train --config configs/train_ensemble_model_lora.yaml &