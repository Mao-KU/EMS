# EMS: Efficient and Effective Massively Multilingual Sentence Embedding Learning
- The paper of this project has been accepted and published by IEEE/ACM Transactions on Audio, Speech, and Language Processing. [IEEE published version](https://ieeexplore.ieee.org/abstract/document/10534791), [arxiv preprint](https://arxiv.org/abs/2205.15744)
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
[1] Zhuoyuan Mao, Chenhui Chu, Sadao Kurohashi. 2024. [*EMS: Efficient and Effective Massively Multilingual Sentence Embedding Learning*](https://arxiv.org/abs/2205.15744)
```
@ARTICLE{10534791,
  author={Mao, Zhuoyuan and Chu, Chenhui and Kurohashi, Sadao},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={EMS: Efficient and Effective Massively Multilingual Sentence Embedding Learning}, 
  year={2024},
  volume={32},
  number={},
  pages={2841-2856},
  keywords={Training;Computational modeling;Task analysis;Data models;Laser modes;Computer architecture;Adaptation models;Efficient and effective multilingual sentence embedding (EMS);cross-lingual token-level reconstruction (XTR);contrastive learning;zero-shot cross-lingual transfer;cross-lingual sentence retrieval;cross-lingual sentence classification},
  doi={10.1109/TASLP.2024.3402064}}
```

