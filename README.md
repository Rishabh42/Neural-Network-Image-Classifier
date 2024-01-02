# Neural Network Image Classifier
**Description:** This project dives into comparing Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs) for image classification using Fashion MNIST and CIFAR-10 datasets. We explore various factors impacting their performance, such as depth, activation functions, and initialization methods.
* The best MLP achieved 88.61% accuracy on Fashion MNIST and 50.11% on CIFAR-10, while the CNN outperformed with 91.4% and 71.7% accuracy, respectively.
* Our discoveries show that CNNs are better for classifying images compared to MLPs.

`report.pdf` contains a description of all our experiments and results.

## Installation
Before running the project, you need to set up the required environment. Follow these steps:

**1. Clone the Repository:**
```
git clone https://github.com/Rishabh42/Neural-Network-Image-Classifier.git
cd Neural-Network-Image-Classifier
```
**2. Create a Virtual Environment (Optional but Recommended):**
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
**3. Install Dependencies:**
```
pip install -r requirements.txt
```

## Usage
To use this project, follow these steps:

**1. Run Jupyter Notebooks:**
* Launch Jupyter Notebook in the project directory:
```
jupyter notebook
```
* Open the relevant Jupyter notebooks, such as:
  - `experiments.ipynb` - contains all of the linear and logistic regression experiments
  - `data_analysis/A2_analysis.ipynb`
  
**2. Explore the Code:**
* Review the codebase:
  - `models/cnn.py` - contains the CNN model
  - `models/mlp.py` - contains the MLP model
  - `utils/` - contains helper code and functions
 
**3. Customize and Experiment:**
* Feel free to customize parameters and experiment with the code.
* Note any additional instructions provided within the notebooks.

