# Temperature Nowcasting GCN — 1-hour Ahead Forecast

This project trains a node-level spatio-temporal model for cities using a Temporal CNN (over lags)
and a GCN over a k-NN city graph. Targets: next-hour temperature (°C) and wind speed (m/s).

## Install
* Donwload and install Python 3.13 from https://www.python.org/downloads/windows/
* Navigate to the project's source folder and open terminal there and run the commands below:
* py -3.13 -m venv .venv
* .\.venv\Scripts\activate
* pip install -r requirements.txt

if the installation (step 5) fails for some reason then run manual installation in terminal:
* python -m pip install --upgrade pip setuptools wheel
* python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.6"
* python -m pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.2 pyyaml==6.0.2 tqdm==4.66.5

## Train
Navigate to the project's source folder and open terminal there and run the commands below:

* .venv\Scripts\python weather_gcn_citynodes.py  --cities .\HistoricalData\ListOfCities.txt --data_dir .\HistoricalData --output_dir .\artifacts --infer_from_checkpoint .\artifacts\best_model.pt --lags 12 --k 6 --epochs 20


## Infer (run with live data)
Navigate to the project's source folder and open terminal there and run the command below:
* .\.venv\Scripts\python weather_gcn_citynodes.py --cities .\HistoricalData\ListOfCities.txt --data_dir .\HistoricalData --output_dir .\artifacts --infer_from_checkpoint .\artifacts\best_model.pt --lags 12 --k 6 --live_nws  --nws_user_agent "my-weather-gnn/1.0 (me@example.com)" --backup_json .\Backup.json

If the process succeed then the folowing files should be created/recreated in "\artifacts" folder:
* A_hat.npy  ⇒ normalized adjacency matrix for GCN
* best_model.pt ⇒ "best" model validation checkpoint (Weights)

## Command params details
* --lags 12 — time window length. We take the last 12 hours for each node, from which the model predicts the next hour. 
More lags ⇒ wider "memory," longer and more difficult training; fewer lags ⇒ faster, but may lose daily patterns. Typical: 6–24.

* -k 6 — number of neighbors in the k-NN graph of cities. Each node has connections with 6 nearest cities. Smaller k ⇒ weaker information exchange, possible "gaps"; larger k ⇒ greater smoothing, the higher the risk of "blurring" local effects. Typical: 3–8.

* -epochs 20 — number of full passes through the training set. Larger ⇒ longer training and a better chance of convergence; too many ⇒ risk of overfitting. We have early stopping, so training may end earlier.

Important: When Infer, the "--lags" and "--k" values ​​must be the same as during training. The "--epochs" only affects training.


## Implementation description
Navigate to the project's source folder and find the Docs folder. This folder contains pdf file with the project description.