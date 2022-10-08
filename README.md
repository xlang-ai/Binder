# Binder
<p align="left">
    <a href="https://img.shields.io/badge/PRs-Welcome-red">
        <img src="https://img.shields.io/badge/PRs-Welcome-red">
    </a>
    <a href="https://img.shields.io/github/last-commit/HKUNLP/Binder?color=green">
        <img src="https://img.shields.io/github/last-commit/HKUNLP/Binder?color=green">
    </a>
    <br/>
</p>

Code for paper [Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875). Please refer to our [demo page]() to have an instant experience of Binder.

<img src="pics/binder.png" align="middle" width="100%">


## Updates

## Dependencies
To establish the environment run this code in the shell:
```bash
conda env create -f py3.7binder.yaml
pip install records==0.5.3
```
That will create the environment `binder` we used.


## Usage

### Environment setup
Activate the environment by running
``````shell
conda activate binder
``````

### Add key
Get `private key`(sk-xxxx like) from OpenAI, and save the key in `key.txt` file, make sure you have the rights to access the model you need.

### Run
Check out commands in `run.py`

## Citation
If you find our work helpful, please cite as
```
@inproceedings{Cheng2022BindingLM,
  title={Binding Language Models in Symbolic Languages},
  author={Zhoujun Cheng and Tianbao Xie and Peng Shi and Chengzu Li and R.K. Nadkarni and Yushi Hu and Caiming Xiong and Dragomir Radev and Marilyn Ostendorf and Luke Zettlemoyer and Noah A. Smith and Tao Yu},
  journal={arXiv preprint arXiv:2210.02875},
  year={2022}
}
```

## Contributors
<a href="https://github.com/BlankCheng">  <img src="https://avatars.githubusercontent.com/u/34505296?v=4"  width="50" /></a> 
<a href="https://github.com/Timothyxxx">  <img src="https://avatars.githubusercontent.com/u/47296835?v=4"  width="50" /></a>
<a href="https://github.com/chengzu-li"><img src="https://avatars.githubusercontent.com/u/69832207?v=4"  width="50" /></a>
<a href="https://github.com/Impavidity">  <img src="https://avatars.githubusercontent.com/u/9245607?v=4"  width="50" /></a> 
<a href="https://github.com/Yushi-Hu"><img src="https://avatars.githubusercontent.com/u/65428713?v=4"  width="50" /></a>
<a href="https://github.com/taoyds"><img src="https://avatars.githubusercontent.com/u/14208639?v=4"  width="50" /></a>


