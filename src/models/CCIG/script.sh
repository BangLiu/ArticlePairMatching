#!bin/bash

# event experiments
python3.6 main.py --data_type "event" --use_gfeatures  # global feature

python3.6 main.py --data_type "event" --use_vfeatures  # local features
python3.6 main.py --data_type "event" --use_vfeatures --use_gcn  # local features + GCN
python3.6 main.py --data_type "event" --use_vfeatures --use_gcn --use_cd  # local community features + GCN

python3.6 main.py --data_type "event" --use_siamese
python3.6 main.py --data_type "event" --use_siamese --use_gcn
python3.6 main.py --data_type "event" --use_siamese --use_gcn --use_cd

python3.6 main.py --data_type "event" --use_vfeatures --use_siamese --use_gcn
python3.6 main.py --data_type "event" --use_vfeatures --use_siamese --use_gfeatures --use_gcn

python3.6 main.py --data_type "event" --use_vfeatures --use_siamese --use_gcn --use_cd
python3.6 main.py --data_type "event" --use_vfeatures --use_siamese --use_gfeatures --use_gcn --use_cd

# story experiments
python3.6 main.py --data_type "story" --use_gfeatures  # global feature

python3.6 main.py --data_type "story" --use_vfeatures  # local features
python3.6 main.py --data_type "story" --use_vfeatures --use_gcn  # local features + GCN
python3.6 main.py --data_type "story" --use_vfeatures --use_gcn --use_cd  # local community features + GCN

python3.6 main.py --data_type "story" --use_siamese
python3.6 main.py --data_type "story" --use_siamese --use_gcn
python3.6 main.py --data_type "story" --use_siamese --use_gcn --use_cd

python3.6 main.py --data_type "story" --use_vfeatures --use_siamese --use_gcn
python3.6 main.py --data_type "story" --use_vfeatures --use_siamese --use_gfeatures --use_gcn

python3.6 main.py --data_type "story" --use_vfeatures --use_siamese --use_gcn --use_cd
python3.6 main.py --data_type "story" --use_vfeatures --use_siamese --use_gfeatures --use_gcn --use_cd

