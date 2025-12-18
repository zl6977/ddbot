# DDBot

This is the repo for the pipeline-based mnemonic recognizer.

## How to run

0. Prepare the environment variable "OPENAI_API_KEY", which is your OpenAI API key.
1. Navigate to `.\server`
2. `pip install -r requirements.txt`
3. `python app.py`
4. Go to the web-based GUI.

## Example data

For simple testing, it is recommended to try a small data file (`data_store\test_data\free_text_input.yaml`).

The data used in the paper is located in `data_store\test_data\Volve open data`:

1. `data_store\test_data\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\1\1\1\00001.xml`
2. `data_store\test_data\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\2\1\1\00001.xml`
3. `data_store\test_data\Volve open data\WITSML Realtime drilling data\Norway-NA-15_$47$_9-F-9 A\1\log\2\2\1\00001.xml`
   They can be used as input in the web-based GUI.

## Evaluations

The evaluations in the paper are in `notebooks\metrics_calculation.ipynb`.
