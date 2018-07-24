This is a simple classifier of portfolio based on deep-neural-network(DNN).
# Description of files
* classifier_2c : classifier with 2 classes
  * DNN_toy_model_with_dense_code : use multi-hot (k-hot) code of portfolio as input features
  * DNN_toy_model_with_factors : use a few tens of factors (based on statistics and techniqual analysis) of portfolio as input features
  * cascading_DNN : use both multi-hot code and factors as input features. Multi-hot code is concatenated with factors after an auto-encoder layer
  * cascading_DNN_sparse_input : same architecture as cascading_DNN with input features are expressed as sparse tensor to save some space
* classifier_4c : classifier with 4 classes
  * same as classifier_2c
