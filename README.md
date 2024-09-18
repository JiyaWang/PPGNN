# BPGNN

## Datasets
`Cora`, `CiteSeer`, `PubMed` and `Photo` are included in the pytorch-geometric package, avilable [here](https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric/data).


## Dependencies
Create a Conda virtual environment and install all the necessary packages

```
conda create --name PPGNNenv python=3.7
conda activate PPGNNenv
```

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install pyg_lib==0.2.0 torch_scatter==2.1.0 torch_sparse==0.6.16 torch_cluster==1.6.0 torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-1.12.0%2Bcu113.html
pip install torch_geometric==2.3.1 
pip install pytorch_lightning==1.9.5
pip install tensorboard
```


## Run
Run the following command:
```
python main.py 
``` 
