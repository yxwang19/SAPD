## SAPD
Here is the comprehensive guide about source code usage of ***A Style-Aware Polytomous Diagnostic Model for Individual Traits***(AKA SAPD)
## Requirements
Before run the SAPD,you need to install the following dependencies via pip or conda.
```shell
dgl==1.1.0+cu118
numpy==2.2.5
pandas==2.2.3
scikit_learn==1.6.1
scipy==1.15.2
torch==2.0.0+cu118
tqdm==4.67.1
swanlab=0.6.6
numpy==1.23.5
pandas==1.5.2
EduCDM==0.0.13
```
## Usage guide
To run the SAPD,you need to enter the SAPD directory.  
Unzip SAPD.zip, enter the directory named SAPD, and then execute the command in the terminal.  
The following is a reference command. If you interested in other parameter of command,you can look into the ***main.py*** for more detail.
```shell
# Run on Suzhou dataset
python main.py --exp_type=cdm --method=sapd --datatype=OECDSuzhou --test_size=0.2 --seed=0 --device=cuda:0 --epoch=10 --batch_size=1024 --lr=0.003 --option_num=5
# Run on Houston dataset
python main.py --exp_type=cdm --method=sapd --datatype=OECDHouston --test_size=0.2 --seed=0 --device=cuda:0 --epoch=10 --batch_size=1024 --lr=0.003 --option_num=5
# Run on Moscow dataset
python main.py --exp_type=cdm --method=sapd --datatype=OECDMoscow --test_size=0.2 --seed=0 --device=cuda:0 --epoch=10 --batch_size=1024 --lr=0.003 --option_num=5
# Run on BIG5 dataset
python main.py --exp_type=cdm --method=sapd --datatype=BIG5 --test_size=0.2 --seed=0 --device=cuda:0 --epoch=10 --batch_size=1024 --lr=0.003 --option_num=5
# Run on EQSQ dataset
python main.py --exp_type=cdm --method=sapd --datatype=EQSQ --test_size=0.2 --seed=0 --device=cuda:0 --epoch=10 --batch_size=1024 --lr=0.003 --option_num=4
```
## Experiment
We use wandb to visualization our experiment result.  
If you prefer not to use it,you can add --wandb=False in your command to disable wandb.
