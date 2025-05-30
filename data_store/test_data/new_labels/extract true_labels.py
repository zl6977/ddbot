import pandas as pd
import os

def extract_data_from_excel(file_path, sheet_name=None, columns_to_extract=None, output_dir='data_store/test_data/new_labels/extracted_json'):
    """
    Extracts data from specified sheet or all sheets of an Excel file,
    optionally selects specific columns, and saves to JSON if a sheet name is specified.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str, optional): The name of the sheet to extract. If None, all sheets are extracted.
        columns_to_extract (list, optional): A list of column names to extract.
                                              Only applicable if sheet_name is provided.
        output_dir (str): Directory to save JSON files.

    Returns:
        dict: A dictionary where keys are sheet names and values are pandas DataFrames.
              If a specific sheet_name is provided, the dictionary will contain only that sheet.
              Returns None on error.
    """
    try:
        if sheet_name:
                   df = pd.read_excel(file_path, sheet_name=sheet_name)
                   print(f"Successfully read sheet: {sheet_name}")
                   print(f"Columns in sheet '{sheet_name}': {df.columns.tolist()}")

                   if columns_to_extract:
                       # Ensure columns exist before selecting
                       missing_columns = [col for col in columns_to_extract if col not in df.columns]
                       if missing_columns:
                           print(f"Warning: The following columns were not found in sheet '{sheet_name}': {missing_columns}")
                           # Filter out missing columns and proceed with existing ones
                           columns_to_extract = [col for col in columns_to_extract if col in df.columns]
                           if not columns_to_extract:
                               print(f"Error: No valid columns to extract after filtering missing ones for sheet '{sheet_name}'.")
                               return None

                       df = df[columns_to_extract]
                       # Rename the newly extracted columns as requested
                       rename_mapping = {'true label': 'true label PrototypeData', 'true label.1': 'true label Unit', 'true label.2': 'true label Quantity'}
                       
                       # Create a list of current column names to iterate over for renaming
                       current_columns = df.columns.tolist()
                       
                       # Apply renaming to the DataFrame
                       df = df.rename(columns=rename_mapping)
                       
                       # Check for duplicate column names after renaming
                       if len(set(df.columns.tolist())) != len(df.columns.tolist()):
                           print("Error: Duplicate column names detected after renaming. Adjust rename_mapping to ensure unique names.")
                           return None
                       
                       print(f"Selected and renamed columns. Current columns: {df.columns.tolist()}")

                   # Save to JSON if a specific sheet is being processed
                   os.makedirs(output_dir, exist_ok=True)
                   json_file_path = os.path.join(output_dir, f"{sheet_name}.json")
                   df.to_json(json_file_path, orient='records', indent=4)
                   print(f"Data from sheet '{sheet_name}' saved to '{json_file_path}'")
                   return {sheet_name: df}
        else:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            data = {}
            for s_name in sheet_names:
                df = pd.read_excel(xls, sheet_name=s_name)
                data[s_name] = df
                print(f"Successfully read sheet: {s_name}")
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except ValueError as ve:
        print(f"Error: Sheet '{sheet_name}' not found in the Excel file. {ve}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return None

if __name__ == "__main__":
    excel_file_path = 'data_store/test_data/new_labels/new_labels.xlsx'

    # Example 1: Extract all sheets
    print("--- Extracting all sheets ---")
    all_sheets_data = extract_data_from_excel(excel_file_path)
    if all_sheets_data:
        print("\nData extracted from all sheets:")
        for sheet_name, df in all_sheets_data.items():
            print(f"\n--- Sheet: {sheet_name} ---")
            print(df.head()) # Print first 5 rows of each DataFrame

    # Example 2: Extract specific columns from a specific sheet and save to JSON
    print("\n--- Extracting specific columns from a specific sheet and saving to JSON ---")
    # You might need to inspect new_labels.xlsx to find actual sheet names and column headers.
    # For demonstration, let's assume there's a sheet named 'Sheet1' and columns 'A', 'B', 'G', 'P', 'Q'.
    specific_sheet_name_1 = '9-F-9_A+1+log+1+1+1+00001' # Updated with an actual sheet name from your Excel file
    specific_sheet_name_2 = '9-F-9_A+1+log+2+1+1+00001'
    specific_sheet_name_3 = '9-F-9_A+1+log+2+2+1+00001'
    columns_to_save = ['Mnemonic', 'Description in data log', 'Unit', 'DataType', 'dataSource', 'true label', 'true label.1', 'true label.2'] # Updated with actual column names from Excel output

    print(f"\n--- Processing sheet: {specific_sheet_name_1} ---")
    specific_sheet_data_json_1 = extract_data_from_excel(excel_file_path, sheet_name=specific_sheet_name_1, columns_to_extract=columns_to_save)
    if specific_sheet_data_json_1:
        print(f"\nData from sheet '{specific_sheet_name_1}' (columns {columns_to_save}) processed.")

    print(f"\n--- Processing sheet: {specific_sheet_name_2} ---")
    specific_sheet_data_json_2 = extract_data_from_excel(excel_file_path, sheet_name=specific_sheet_name_2, columns_to_extract=columns_to_save)
    if specific_sheet_data_json_2:
        print(f"\nData from sheet '{specific_sheet_name_2}' (columns {columns_to_save}) processed.")

    print(f"\n--- Processing sheet: {specific_sheet_name_3} ---")
    specific_sheet_data_json_3 = extract_data_from_excel(excel_file_path, sheet_name=specific_sheet_name_3, columns_to_extract=columns_to_save)
    if specific_sheet_data_json_3:
        print(f"\nData from sheet '{specific_sheet_name_3}' (columns {columns_to_save}) processed.")
