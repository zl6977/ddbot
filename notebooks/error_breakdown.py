import json
import pandas as pd

# Read the original JSON file with 3 evaluation sets
with open("notebooks/evaluation/labels_logprob_dif_llm.json", "r") as f:
    data = json.load(f)

print("Original data loaded successfully!")
print(f"Original data: {len(data)} evaluation sets")

# Merge the 3 evaluation sets into single category dictionaries
merged_data = {}

categories = ['Quantity', 'Unit', 'PrototypeData']
fields_to_merge = ['true', 'pred', 'conf', 'query_difficulty_score', 'llm_judge_score']

for category in categories:
    merged_data[category] = {}

    for field in fields_to_merge:
        # Concatenate all lists for this field across the 3 evaluation sets
        merged_list = []
        for eval_set in data:
            if category in eval_set and field in eval_set[category]:
                merged_list.extend(eval_set[category][field])

        merged_data[category][field] = merged_list

print("\nMerged data structure:")
for category in categories:
    print(f"{category}: {len(merged_data[category]['true'])} total samples")

# Save the merged data to JSON file
merged_json_path = "notebooks/evaluation/labels_logprob_dif_llm_merged.json"
with open(merged_json_path, "w") as f:
    json.dump(merged_data, f, indent=2)

print(f"\nMerged data saved to: {merged_json_path}")

# Create Excel file with multiple sheets
output_path = "notebooks/evaluation/merged_evaluation_data.xlsx"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    categories = ['Quantity', 'Unit', 'PrototypeData']

    for category in categories:
        # Create DataFrame for this category
        df = pd.DataFrame({
            'true': merged_data[category]['true'],
            'pred': merged_data[category]['pred'],
            'conf': merged_data[category]['conf'],
            'query_difficulty_score': merged_data[category]['query_difficulty_score'],
            'llm_judge_score': merged_data[category]['llm_judge_score']
        })

        # Convert lists to strings for better Excel display
        df['true'] = df['true'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        df['pred'] = df['pred'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

        # Write to Excel sheet
        df.to_excel(writer, sheet_name=category, index=False)

        print(f"{category} sheet: {len(df)} rows exported")

print(f"\nExcel file created successfully: {output_path}")
print("Sheets created: Quantity, Unit, PrototypeData")
print("Each sheet contains columns: true, pred, conf, query_difficulty_score, llm_judge_score")
