# Machine Learning Summer School Tutorial
Repository for the Machine learning summer school tutorial 

#Tutorial 
Decoding the Invisible: A Comprehensive Guide to Understanding and Overcoming Limitations of explanations in Radiology AI Image Interpretation

---

## Virtual Environment Setup

1. Make sure Python 3 is installed on your system. You can check the version by running the following command:
   ```bash python3 --version``` <br/><br/>

2. stall virtualenv if you don't have it installed already. You can install it using pip:
    ```python3 -m pip install --user virtualenv```  <br/><br/>

3. Create a new virtual environment for the project:
    ```python3 -m venv env``` <br/><br/>

4. Activate the virtual environment:
        For Linux/macOS:
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ```source env/bin/activate``` 
        For Windows (PowerShell):
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ```.\env\Scripts\Activate.ps1```
        For Windows (Command Prompt):
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ```.\env\Scripts\activate.bat``` <br/><br/>

5. Install project dependencies by running the following command in the project directory:
    ```pip install -r requirements.txt``` <br/><br/>

6. You're now ready to run the project within the virtual environment.


---
---
# Dataset
The dataset used in this project is the Kaggle - Pneumonia detection dataset. you can find it [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
Once downloaded, place the zip file under the data directory of the project folder and unzip it. The data directory will then look like what is shown in the Directory Structure below.

---
---
## Directory Structure

mlss
├── README.md
├── requirements.txt
├── data/
│   ├── chest_xray/
│   │   ├── train
│   │   │   ├── NORMAL
│   │   │   ├── PNEUMONIA
│   │   ├── test
│   │   │   ├── NORMAL
│   │   │   ├── PNEUMONIA
│   │   ├── valid
│   │   │   ├── NORMAL
│   │   │   ├── PNEUMONIA
├── logs/
├── models/
├── notebooks/
├── outputs/
├── src/
│   ├── run.py - This script merges all the other scripts and starts the training once executed
│   ├── training.py - This Script has the training block. it is executed for every iteration while the model is training in run.py. It is responsible for updating the weights of the model and making it learn
│   ├── validate.py - This Script has the validation block. it is executed for every iteration after the model is trained in run.py. It is used to validate the model after every iteration of a left out dataset (instances of data that the model has not seen while training).
│   ├── dataset.py - This Script has the dataloading code. it is used to read and fetch the data efficiently along with the necessary preprocessing from the storage drive/s. Once an object of this class is created in run.py, it is fed to a pytorch dataloader (it is responsible to load the data efficiently from the storage to the Graphical Processing Unit to make the training faster).
│   ├── config.py - This script contains constants that don't change throughout the scripts such as Image height and width on which the model trains, the data directories, the model and logs path, etc.


---
---
## Visualising the logs
#### After the model is trained and [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) logs are generated, you can visualize in your browser by running the following command in the command line/terminal.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```tensorboard --logdir "-DIRECTORY WHERE THE LOGS ARE-"```
if you are following the same directory structure, the logs should be in mlss/logs/V0/MODEL_NAME/CURRENT_DATE/events.out.tfevents**

Note: 
- MODEL_NAME will be the name you have given in src/run.py in the main block (at the bottom of the script)
- CURRENT_DATE will be the date and time when you execute run.py (the code can be found in src/config.py)
- V0 is the version, you can change if from the EXP_NAME from src/config.py. It is used to not confuse multiple versions of a trained model.

---

## Checking the trained model weights
#### The trained model weights can be found at mlss/models/V0/MODEL_NAME/CURRENT_DATE/

The mentioned directory will have multiple files as listed below
- best_model_config.pth.tar (This is the best model weights) </br></br>
- epoch_20.pth.tar
- epoch_40.pth.tar
- epoch_60.pth.tar
- epoch_70.pth.tar
The above are the weights saved at every 20 iterations, the reason to have these is that if the system crashes or some unusual interuption in the model training arrises, one can load the weights and continue the training. 
Note: The frequency for this can be change in src/config.py by changing the value of SAVE_WEIGHTS_INTERVAL