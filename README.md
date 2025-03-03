# Intraoperative Hypotension Prediction

This project aims to develop a deep learning model to predict intraoperative hypotension (IOH) during surgery.

## 📌 Project Structure

- `data/`: Directory for storing data files
- `src/`: Source code directory
  - `data_maker.py`: Script for preprocessing raw data
  - `train.py`: Script for training the model
  - `run.py`: Main execution script
- `models/`: Directory for storing trained models
- `requirements.txt`: List of required Python packages

## 🚀 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/skku-dhkim/intraoperative_hypotension.git
   cd intraoperative_hypotension
   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## 🔬 Usage

1. **Data Preprocessing:**

   ```bash
   python src/data_maker.py
   ```

   This script processes raw physiological data (e.g., arterial blood pressure, ECG) and prepares it for model training.

2. **Model Training:**

   ```bash
   python src/train.py --epochs 50 --batch_size 32 --lr 0.001
   ```

   This will train the model using the preprocessed data.

3. **Run Prediction:**

   ```bash
   python src/run.py --input_file data/test_sample.csv
   ```

   This script loads a trained model and predicts intraoperative hypotension from new patient data.

## 📊 Experimental Results

- The model achieved **high sensitivity and specificity** in predicting IOH episodes.
- Implemented **real-time inference** for potential clinical applications.
- Tested on multiple datasets for **robustness and generalization**.

## 📄 References

- [Predicting Intraoperative Hypotension Using Deep Learning](https://github.com/MediTeamOne/IOH_prediction)
- [Use data-based approach to predict intraoperative hypotension](https://github.com/BobAubouin/hypotension_pred)
- [The Hypotension Prediction Index (HPI) validation study](https://github.com/crph-utwente/HPIvalidation)

## ⚖ License

This project is released under the MIT License.
