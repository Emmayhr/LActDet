# LActDet
## Inroduction
This is a prototype demo for attack activity detection, which inputs the alert logs in .csv format and outputs the activity type.
## Categoty Decription
* checkout: This dir is used to restore parameters of se2seq and classifier model.
* data: This dir stores the alert logs in .csv format.
* dataset: We restore the vectorized representation of activity in this dir.
* supervised_model: This dir includes event extraction, phase embedder, classifier codes.
* test: It stores intermediate process files.
* log: This dir includes debugging information.
## Preparing
Before running, python3, torch, scikit-learn, numpy, scipy, pandas, and tqdm need to be installed.
## Running
cd LActdet
python3 supervised_model/main64.py --method LActDet/e-LActDet/LWS

