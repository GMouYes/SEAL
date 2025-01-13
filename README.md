# SEAL
Code Repository for PerCom 2025 paper SEAL "Semantically Encoding Activity Labels for Context-Aware Human Activity Recognition" published in [Percom](https://www.percom.org/)

## Source Code SEAL
We are excited to share our latest work on improving HAR performance with a language model to encode the semantic information of context and activity labels. Down below we share the detailed runnable code and configuration template with sampled data (2 batch sliced example).

## What is in the repo
- bash script in ``code`` folder for running the code ``train.sh``
- folder ``code`` containing all python code
- folder ``config`` containing the configuration file, indicating the best hyperparameters we used for Extrasensory. You can set search ranges for Gaussian optimization for other datasets.
- folder ``data`` containing sampled data (not the full data)

We showcase an example of running the given code on a sampled data slice. [Extrasensory](http://extrasensory.ucsd.edu/#paper.vaizman2017a) is a public dataset and researchers should be able to download and process the full original source dataset (unfortunately, we do not own this dataset). For more details, please refer to their original paper.

## How to run the code
- Make sure the required packages are installed with compatible versions. 
- Unzip folders under data (even the sampled users with sampled example files are large)
- Modify the ``MLP_BERT_extra_ray_config.yml`` file hyper-parameter settings
- Run the script with ``train.sh``
- Check printline logs in ``log`` folder and the results in ``output`` folder
- Change the checkpoint path to the best checkpoint and evaluate model's performance with script ``test.sh``
