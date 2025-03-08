#!/usr/bin/bash
scp -r requirements.txt src *.py bin aorus:projects/brave-new-dolomiti/

# scp -r data/data_dolomiti.npz data/data_kaggle.npz data/data_merged_dolomiti_kaggle aorus:projects/brave-new-dolomiti/data/

# scp -r data/data_dolomiti.npz aorus:projects/brave-new-dolomiti/data/
