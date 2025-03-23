Code for “ Distribution-Aware Unsupervised Attacks on Graph Contrastive Learning”

Requirements：see requirements.txt

train
python myattack.py --dataset Cora

eval node classification
python train_GCA.py --dataset Cora --perturb --attack_method PPP --attack_rate 0.10 --device cuda:0

eval link prediction
python train_LP.py --dataset Cora --perturb --attack_method PPP --attack_rate 0.10 --device cuda:0
