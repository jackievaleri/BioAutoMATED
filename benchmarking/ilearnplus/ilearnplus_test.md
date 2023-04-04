Since the web version of iLearnPlus has a maximum of 2000 sequences, I downloaded iLearnPlus, including its AutoML module, from [this link](https://github.com/Superzchen/iLearnPlus/).


I got the environment working with the following:
```conda create --name ilearnplus
conda activate ilearnplus
conda install pytorch torchvision torchaudio -c pytorch
pip3 install ilearnplus
pip3 install PyQt5 matplotlib seaborn
python
	>>> from ilearnplus import runiLearnPlus
	>>> runiLearnPlus()
```

This generates a GUI which I then accessed to generate the benchmarking results as described in the methods section.