```
usage: train.py [-h] [--model_name MODEL_NAME] [--model_type MODEL_TYPE] [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--model_ckpt MODEL_CKPT] [--output_path OUTPUT_PATH] [--num_epochs NUM_EPOCHS] 
                [--rdrsegmenter_path RDRSEGMENTER_PATH] [--lr LR] [--dropout DROPOUT] [--increment_dropout_prob INCREMENT_DROPOUT_PROB] [--train_folds TRAIN_FOLDS [TRAIN_FOLDS ...]]
                [--scheduler_type SCHEDULER_TYPE] [--batch_size BATCH_SIZE] [--seed SEED] [--is_smoothing IS_SMOOTHING] [--num_folds NUM_FOLDS] [--optimizer_type OPTIMIZER_TYPE] [--preprocess PREPROCESS]

Argument Parser for Training model

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        select model name
  --model_type MODEL_TYPE
                        select model type
  --train_path TRAIN_PATH
                        training df path
  --test_path TEST_PATH
                        test df path
  --model_ckpt MODEL_CKPT
                        path to model ckpt folder
  --output_path OUTPUT_PATH
                        path to output folder
  --num_epochs NUM_EPOCHS
                        change number of epochs
  --rdrsegmenter_path RDRSEGMENTER_PATH
                        rdrsegmenter path
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
                        is smoothing or not
  --num_folds NUM_FOLDS
                        number of folds
  --optimizer_type OPTIMIZER_TYPE
                        choose optimizer type, group or basic
  --preprocess PREPROCESS
                        replace word or not
```