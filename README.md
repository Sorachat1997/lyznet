# lyznet 

## Description

A lightweight python tool for training and verifying neural **Ly**apunov functions by solving **Z**ubov's partial differential equation (PDE) using physicis-informed neural **net**works (PINN). It is built upon PyTorch for training neural networks and the numerical satisifability module theories (SMT) solver dReal (https://github.com/dreal/dreal4) for verification. The main functionality is to perform stability analysis using neural Lyapunov functions and provide region-of-attraction estimates in terms of sublevel sets of these neural Lyapunov functions. The results are formally verified by the SMT solver dReal. 

## Installation

Installation and testing were conducted using Python 3 (version 3.7 or higher).

1. **Install the project:**
    Navigate to the root folder of this project and run the following command to install it locally:
    ```bash 
    pip install -e .
    ```
    
2. **Install dreal4:**
   Follow the instructions [here](https://github.com/dreal/dreal4) to install dreal4, including its python binding. 

3. **Install other required packages:**
   You can install the following Python packages using pip:
   ```bash
   pip install torch numpy scipy sympy matplotlib joblib tqdm
   ```
   
## Usage

You can run any example file directly using Python. For example, to run `example.py`, use the following command:
```bash
python example.py
```
