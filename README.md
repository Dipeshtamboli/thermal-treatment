# Jue-Mo Thermal-Treatment-Programm

## 1. Data cleaning
Run `$python3 clean_data.py` command after changing the experiment name to choose the correct data file.

## 2. Model training 
Run `python3 train.py` to train a model. The code will output a `.pt` file like `Oak_model.pt` and will used for inference.

## 3. Inference and Plotting
Run `python3 inference.py` to generate R plots. Change plot colour, box colour, legend, etc. from here.