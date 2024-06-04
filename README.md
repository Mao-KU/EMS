# EMS: Efficient and Effective Massively Multilingual Sentence Embedding Learning
- The paper of this project has been accepted and published by IEEE/ACM Transactions on Audio, Speech, and Language Processing. [IEEE published version](https://ieeexplore.ieee.org/abstract/document/10534791) [arxiv preprint (EMS: Efficient and Effective Massively Multilingual Sentence Representation)](https://arxiv.org/abs/2205.15744)
- This is the codebase for training multilingual sentence embeddings.
- Currently our EMS model supports 62 languages.
- You can do inference for sentences with our pre-trained EMS model (refer to Inference below).
- Or you can also efficiently train a new multilingual sentence embedding model for your interested languages (refer to Train from Scratch below).

## Prerequisites:
```
conda env create --file ems.yaml
```
If you use A100 GPU cards, use
```
conda env create --file ems_a100.yaml
```
According to specific cuda version, you might have to reinstall a proper torch version.

## Get Started:
```
git clone https://github.com/Mao-KU/EMS.git
cd EMS
```
Download our pre-trained model:
```
mkdir -p ckpt
cd ckpt
wget -c https://lotus.kuee.kyoto-u.ac.jp/~zhuoyuanmao/EMS_model.pt
cd ..
```
Do inference for a file with one English sentence per line.
```
cd inference
python embed.py -l en --data_path eng_sample --gpu --save_to eng_sample.pkl
```
Now eng\_sample with one English sentence has been encoder into a numpy array with the shape of (1, 1024), which is saved as a binary file "inference/eng\_sample.pkl". Refer to the following for more details about inference and model training.

## Inference:
### Preparations:
- DATA: one sentence per line.

### Inference with command lines and save to a binary file:
- Inference with our pre-trained model:
```
cd inference
python embed.py -l ${LANGUAGE_OF_YOUR_DATA} --data_path ${PATH_OF_DATA} --gpu --save_to ${PATH_TO_SAVE_NUMPY_PICKLE_FILE}
```
- Inference with your from-scratch model:
```
cd inference
python embed.py -l ${LANGUAGE_OF_YOUR_DATA} --data_path ${PATH_OF_DATA} -m ${MODEL_NAME} --model_path ${MODEL_PATH} --gpu --save_to ${PATH_TO_SAVE_NUMPY_PICKLE_FILE}
```
Note: Define your "model name" and model architectures in "inference/config.py". "model name" is set as "EMS" by default.
- The encoded embeddings are with the numpy shape of (number of the sentences, 1024) and saved to be a pickle file, which can be reloaded in a python file as follows:
```
import pickle
f = open(PATH, "rb")
# PATH denotes the path to your embeddings
# same as the value of --save_to above
embs = pickle.load(f)
# ... Your code here.
```
### Inference in python: Refer to "inference/eval.py", which imports "inference/embed.py" as a python library.
```
python eval.py -l ${LANGUAGE_OF_YOUR_DATA} --data_path ${PATH_OF_DATA} --gpu
```

## Train from Scratch:
TO BE UPDATED.

## Reference
[1] Zhuoyuan Mao, Chenhui Chu, Sadao Kurohashi. 2022. [*EMS: Efficient and Effective Massively Multilingual Sentence Representation Learning*](https://arxiv.org/abs/2205.15744)
```
@article{DBLP:journals/corr/abs-2205-15744,
  author    = {Zhuoyuan Mao and
               Chenhui Chu and
               Sadao Kurohashi},
  title     = {{EMS:} Efficient and Effective Massively Multilingual Sentence Representation
               Learning},
  journal   = {CoRR},
  volume    = {abs/2205.15744},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.15744},
  doi       = {10.48550/arXiv.2205.15744},
  eprinttype = {arXiv},
  eprint    = {2205.15744},
  timestamp = {Wed, 01 Jun 2022 13:56:25 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-15744.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

