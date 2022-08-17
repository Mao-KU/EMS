# EMS (Efficient and Effective Massively Multilingual Sentence Representation)
- Code for [paper](https://arxiv.org/abs/2205.15744)
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

## Inference:
- Inference with our pre-trained model:
```
cd evaluation
python embed.py -l {LANGUAGE OF YOUR DATA} --data_path {PATH OF DATA} --gpu --save_to {PATH TO SAVE NUMPY PICKLE FILE}
```
The encoded embeddings are with the numpy shape of (number of the sentences, 1024) and saved to be a pickle file, which can be reloaded with "pickle.load()".
- Inference with your from scratch model:
TO BE UPDATED.

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

