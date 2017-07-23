# Knowledge-Guided CVAE for dialog generation

We provide a TensorFlow implementation of the CVAE-based dialog model described in
**Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders**.
See the [paper](https://arxiv.org/abs/1703.10960) for more details.

# Prerequisites
TensorFlow 1.0+
Python 2.7
Numpy

- You may need to pip install beeprint if the module is missing

# Usage
## Train a new model
    python kgcvae_swda.py
will run default training and save model to ./working

## Test a existing model
Modify the TF flags at the top of kgcvae_swda.py as follows to run a existing model

    forward_only: False -> True
    test_path: set to the folder contains the model. E.g. runxxxx
Then you can run the model by:

    python kgcvae_swda.py
The outputs will be printed to stdout and generated responses will be saved at test.txt in the test_path.

## Use pre-trained Word2vec
Download Glove word embeddings from https://nlp.stanford.edu/projects/glove/
Set the path to word embeddings at line 15 of kgcvae_swda.py: word2vec_path.

# References 
If you use any source codes or datasets included in this toolkit in your
work, please cite the following paper. The bibtex are listed below:
 
    [Zhao et al, 2017]:
     @article{zhao2017learning,
      title={Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders},
      author={Zhao, Tiancheng and Zhao, Ran and Eskenazi, Maxine},
      journal={arXiv preprint arXiv:1703.10960},
      year={2017}
    }
