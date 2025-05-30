import json
import pandas as pd


def merge_evaluation_data(input_json_path, merged_json_path, excel_path,
                         categories=None, fields_to_merge=None,
                         include_conf_and_scores=True):
    """
    Merge evaluation data from multiple evaluation sets and export to JSON and Excel.

    Args:
        input_json_path (str): Path to the input JSON file containing evaluation data
        merged_json_path (str): Path to save the merged JSON data
        excel_path (str): Path to save the Excel file
        categories (list): List of categories to process (default: ['Quantity', 'Unit', 'PrototypeData'])
        fields_to_merge (list): List of fields to merge (default: ['true', 'pred', 'conf', 'query_difficulty_score', 'llm_judge_score'])
        include_conf_and_scores (bool): Whether to include conf, query_difficulty_score, llm_judge_score in Excel (default: True)
    """
    if categories is None:
        categories = ['Quantity', 'Unit', 'PrototypeData']
    if fields_to_merge is None:
        fields_to_merge = ['true', 'pred', 'conf', 'query_difficulty_score', 'llm_judge_score']

    # Read the original JSON file
    with open(input_json_path, "r") as f:
        data = json.load(f)

    print(f"Original data loaded successfully from {input_json_path}!")
    print(f"Original data: {len(data)} evaluation sets")

    # Merge the evaluation sets into single category dictionaries
    merged_data = {}

    for category in categories:
        merged_data[category] = {}

        for field in fields_to_merge:
            # Concatenate all lists for this field across the evaluation sets
            merged_list = []
            for eval_set in data:
                if category in eval_set and field in eval_set[category]:
                    merged_list.extend(eval_set[category][field])

            merged_data[category][field] = merged_list

    print("\nMerged data structure:")
    for category in categories:
        print(f"{category}: {len(merged_data[category]['true'])} total samples")

    # Save the merged data to JSON file
    with open(merged_json_path, "w") as f:
        json.dump(merged_data, f, indent=2)

    print(f"\nMerged data saved to: {merged_json_path}")

    # Create Excel file with multiple sheets
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for category in categories:
            # Create DataFrame for this category
            df_data = {
                'true': merged_data[category]['true'],
                'pred': merged_data[category]['pred']
            }

            if include_conf_and_scores:
                df_data.update({
                    'conf': merged_data[category]['conf'],
                    'query_difficulty_score': merged_data[category]['query_difficulty_score'],
                    'llm_judge_score': merged_data[category]['llm_judge_score']
                })

            df = pd.DataFrame(df_data)

            # Convert lists to strings for better Excel display
            df['true'] = df['true'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
            df['pred'] = df['pred'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

            # Write to Excel sheet
            df.to_excel(writer, sheet_name=category, index=False)

            print(f"{category} sheet: {len(df)} rows exported")

    print(f"\nExcel file created successfully: {excel_path}")
    print(f"Sheets created: {', '.join(categories)}")
    columns = ['true', 'pred']
    if include_conf_and_scores:
        columns.extend(['conf', 'query_difficulty_score', 'llm_judge_score'])
    print(f"Each sheet contains columns: {', '.join(columns)}")


# logprob_dif_llm
merge_evaluation_data(
    input_json_path="notebooks/evaluation/labels_logprob_dif_llm.json",
    merged_json_path="notebooks/evaluation/labels_logprob_dif_llm_merged.json",
    excel_path="notebooks/evaluation/merged_evaluation_data_logprob_dif_llm.xlsx",
    include_conf_and_scores=True
)

# DirectRAG
merge_evaluation_data(
    input_json_path="notebooks/evaluation/labels_simpleRAG.json",
    merged_json_path="notebooks/evaluation/labels_directRAG_merged.json",
    excel_path="notebooks/evaluation/merged_evaluation_data_directRAG.xlsx",
    include_conf_and_scores=False
)
