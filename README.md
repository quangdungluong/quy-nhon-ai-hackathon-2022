# QUY NHON AI HACKATHON 2022: CHALLENGE 2 - REVIEW ANALYTICS
## Team CTA. Gà con - Top 2 Solution
## Member:
1. Trần Phan Quốc Đạt - [TPQDat](https://github.com/TPQDat)
2. Đỗ Thành Thông - [DThanhThong](https://github.com/DThanhThong)
3. Lương Quang Dũng - [quangdungluong](https://github.com/quangdungluong)

## Technology:
- Ensemble, ensemble, ensemble
- Pseudo-labelling
- PhoBert
- Label Smoothing
- Cross-Validation

## Preprocessing:
- None. That's NLP in 2022

## Usage
```
usage: train.py [-h] [--model_name MODEL_NAME] [--model_type MODEL_TYPE] [--train_path TRAIN_PATH] [--model_ckpt MODEL_CKPT] [--output_path OUTPUT_PATH] [--rdrsegmenter_path RDRSEGMENTER_PATH]
                [--num_epochs NUM_EPOCHS] [--lr LR] [--dropout DROPOUT] [--increment_dropout_prob INCREMENT_DROPOUT_PROB] [--train_folds TRAIN_FOLDS [TRAIN_FOLDS ...]] [--scheduler_type SCHEDULER_TYPE]
                [--batch_size BATCH_SIZE] [--seed SEED] [--is_smoothing IS_SMOOTHING] [--num_folds NUM_FOLDS] [--optimizer_type OPTIMIZER_TYPE] [--preprocess PREPROCESS]
                [--smoothing SMOOTHING [SMOOTHING ...]] [--num_warmup_steps NUM_WARMUP_STEPS]

Argument Parser for Training model

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        select model name
  --model_type MODEL_TYPE
                        select model type
  --train_path TRAIN_PATH
                        training df path
  --model_ckpt MODEL_CKPT
                        path to model ckpt folder
  --output_path OUTPUT_PATH
                        path to output folder
  --rdrsegmenter_path RDRSEGMENTER_PATH
                        rdrsegmenter path
  --num_epochs NUM_EPOCHS
                        change number of epochs
  --lr LR               change learning rate
  --dropout DROPOUT     change hidden dropout probability
  --increment_dropout_prob INCREMENT_DROPOUT_PROB
                        increment_dropout_prob
  --train_folds TRAIN_FOLDS [TRAIN_FOLDS ...]
                        choose train folds
  --scheduler_type SCHEDULER_TYPE
                        choose scheduler types
  --batch_size BATCH_SIZE
                        choose batch size
  --seed SEED           choose seed
  --is_smoothing IS_SMOOTHING
                        is using label smoothing or not
  --num_folds NUM_FOLDS
                        number of folds
  --optimizer_type OPTIMIZER_TYPE
                        choose optimizer type, group or basic
  --preprocess PREPROCESS
                        replace word or not
  --smoothing SMOOTHING [SMOOTHING ...]
                        choose smoothing params
  --num_warmup_steps NUM_WARMUP_STEPS
                        number of warmup steps in scheduler
```