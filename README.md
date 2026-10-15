# Fake News Detection System — Streamlit

This is a VS Code version of the notebook project. It uses the same core pipeline:
- ISOT Fake and Real News Dataset (`Fake.csv`, `True.csv`)
- Title + article text combined into one content field
- Empty articles and duplicate content removed
- 80/20 stratified train/test split
- TF-IDF vectorizer with unigrams and bigrams
- Logistic Regression classifier
- Accuracy, FAKE precision, FAKE recall, FAKE F1-score, classification report, and confusion matrix
- Streamlit user interface for predicting FAKE/REAL and showing model probabilities

## 1. Install Python

Install Python 3.10 or later, then open this folder in VS Code. In VS Code, install the **Python** extension.

## 2. Create a virtual environment

### Windows PowerShell
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use Command Prompt instead:
```bat
.venv\Scripts\activate.bat
```

## 3. Add the dataset

Download the Kaggle **ISOT Fake and Real News Dataset**:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Put these two files directly inside the `data` folder:
- `data/Fake.csv`
- `data/True.csv`

Do not put the CSV files in an extra nested folder.

## 4. Train the model

Run in the VS Code terminal, from this project folder:
```bash
python train_model.py
```

Training can take a few minutes. On success, the model and evaluation files will be saved in `models/`.

## 5. Run the Streamlit app

```bash
streamlit run app.py
```

Streamlit should open the app in your browser. If it does not, use the local URL printed in the terminal (usually `http://localhost:8501`).

## Troubleshooting
- **`streamlit` is not recognized:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **Dataset not found:** check that both CSV files are inside the `data` folder and spelled exactly `Fake.csv` and `True.csv`.
- **Model not found:** run `python train_model.py` successfully before launching the app.
- **App changes not showing:** save the file and refresh the browser; restart with `Ctrl+C` then `streamlit run app.py` if needed.

## Important limitation
The model identifies patterns learned from this dataset. A prediction is not independent verification of whether a news story is factually true.
