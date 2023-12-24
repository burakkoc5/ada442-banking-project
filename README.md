# Bank Marketing Prediction Project

This project was developed as part of the ADA442 course at TED University. It involves training an XGBoost model on a bank marketing dataset and making predictions using the trained model with a Streamlit interface.

## Installation

You can install project dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Run `main.py` to start the Streamlit application:

```bash
streamlit run main.py
```

2. Once the Streamlit application starts, fill out the form that will appear in your web browser.

3. Click the "Make Prediction" button to prompt the model to make predictions.

## Files

- `requirements.txt`: File containing the dependencies for the project.
- `model.py`: Python script where the XGBoost model is trained and saved.
- `main.py`: Main file containing the Streamlit application.
- `model.pkl`: Pre-trained XGBoost model. To use this model, load it in your Python script using `joblib.load("model.pkl")`.

## Contributors

- Burak Koç
- Furkan Safa Altunyuva
- Buse Öner
- Mehmet Demir

## Course Information

This project was developed as part of the ADA442 course at TED University.

## License

This project is licensed under the [MIT License](LICENSE).
```
