# Introduction
This is the project repo for our paper: Event Knowledge Incorporation with Posterior Regularization for Event-Centric Question Answering, which can be found: https://arxiv.org/abs/2305.04522. 

# Models
## I. Install packages.
We list the packages in our environment in env.yml file for your reference.

## II. Train and test
### 1. Fine-tuned models: will be released upon paper acceptance.

### 2. Train from scratch
Run `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
Run `bash ./code/run_span_pred.sh` and `bash ./code/run_ans_generation.sh`.

### 3. Test on dev set
Run `bash ./code/eval_span_pred.sh` and `bash ./code/eval_ans_generation.sh`.
