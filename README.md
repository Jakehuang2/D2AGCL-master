Code for “ Distribution-Aware Unsupervised Attacks on Graph Contrastive Learning”

Requirements：see requirements.txt


python myattack.py --dataset Cora
python train_GCA.py --dataset Cora --perturb --attack_method PPP --attack_rate 0.10 --device cuda:0
python train_LP.py --dataset Cora --perturb --attack_method PPP --attack_rate 0.10 --device cuda:0
