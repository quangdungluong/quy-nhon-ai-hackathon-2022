```
usage: train.py [-h] [--model_name MODEL_NAME] [--model_type MODEL_TYPE] [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--infer INFER]
                [--model_ckpt MODEL_CKPT] [--submission SUBMISSION] [--output_path OUTPUT_PATH] [--num_epochs NUM_EPOCHS] [--lr LR]       
                [--dropout DROPOUT]

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
  --infer INFER         infer after trained model or not, default=True
  --model_ckpt MODEL_CKPT
                        path to model ckpt folder
  --submission SUBMISSION
                        path to submission file
  --output_path OUTPUT_PATH
                        path to output folder
  --num_epochs NUM_EPOCHS
                        change number of epochs
  --lr LR               change learning rate
  --dropout DROPOUT     change hidden dropout probability
```