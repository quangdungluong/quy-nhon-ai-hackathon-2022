## Model Architecture
Training data is rather small and NLP models are large transformers. This a recipe for training instability or for overfitting, or both.
* First, using classification heads with 5 levels of dropout.
* Second, using a linear combination of all hidden states with a dropout on the linear coefficients. The dropout forces the model heads to learn from all hidden states and not just the last one.

## Preprocessing
https://www.kaggle.com/code/kyakovlev/preprocessing-bert-public/notebook