import json
from collections import defaultdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: Path) -> list[dict]:
    """Loads JSONL data from a file."""
    data = []
    if not file_path.exists():
        logger.error(f"Error: Data file not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Error decoding JSON on line {line_num}: {line.strip()}. Error: {e}")
                continue
    logger.info(f"Loaded {len(data)} records from {file_path}")
    return data

def clean_and_deduplicate_data(input_file_path: Path, cleaned_output_path: Path, redundant_output_path: Path) -> None:
    """
    Reads the input JSONL file, deduplicates skill records based on 'skill_name',
    merging list-type fields from duplicate entries.
    Writes the cleaned data to one file and the identified redundant/duplicate
    records to another file.

    Note: This function handles exact duplicates of skill_name.
    For fuzzy duplicates (e.g., 'Python' vs 'Python Programming'), you need
    to unify those names in the input data *before* running this script.
    Circular dependencies also need to be resolved manually in the input data.
    """
    records = load_data(input_file_path)
    if not records:
        return

    canonical_records_dict = {} # Stores normalized_skill_name -> canonical record
    redundant_records_list = [] # Stores records identified as duplicates

    original_record_count = len(records)
    
    for record in records:
        skill_name = record.get('skill_name')
        if not skill_name:
            logger.warning(f"Record missing 'skill_name', skipping: {record}")
            continue

        normalized_skill_name = skill_name.lower().strip()

        if normalized_skill_name not in canonical_records_dict:
            # If this is the first time we see this skill, add it as canonical
            canonical_records_dict[normalized_skill_name] = record.copy()
        else:
            # If it's a duplicate, merge its data into the existing canonical record
            # and add the current record to the redundant list
            redundant_records_list.append(record.copy()) # Add the current duplicate record to the redundant list

            existing_record = canonical_records_dict[normalized_skill_name]
            logger.info(f"Merging duplicate skill: '{skill_name}'")

            # Merge list-type fields (prerequisites, complementary_skills, industry_usage)
            for key in ['prerequisites', 'complementary_skills', 'industry_usage']:
                if isinstance(record.get(key), list):
                    if key not in existing_record or not isinstance(existing_record[key], list):
                        existing_record[key] = [] # Initialize if missing or wrong type
                    for item in record[key]:
                        if item not in existing_record[key]:
                            existing_record[key].append(item)
            # For other fields, keeping the value from the first encountered record is the default.
            # Custom merging logic for other fields can be added here if needed.

    cleaned_data = list(canonical_records_dict.values())
    duplicate_count = original_record_count - len(cleaned_data)


    # Ensure output directories exist
    cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)
    redundant_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write cleaned data
    with open(cleaned_output_path, 'w', encoding='utf-8') as f:
        for record in cleaned_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f"Cleaned data saved to: {cleaned_output_path}")

    # Write redundant data
    with open(redundant_output_path, 'w', encoding='utf-8') as f:
        for record in redundant_records_list:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f"Redundant data saved to: {redundant_output_path}")

    logger.info(f"Deduplication complete. Original records: {original_record_count}, Duplicates moved to redundant file: {duplicate_count}, Cleaned records: {len(cleaned_data)}")

def main():
    # Corrected input file path based on user's clarification
    input_jsonl = Path("data/processed/cleaned_skills_data.jsonl") 
    cleaned_output_jsonl = Path("data/processed/cleaned_skills_data.jsonl") # Destination for cleaned data
    redundant_output_jsonl = Path("data/processed/redundant_skills_data.jsonl") # Destination for redundant data

    clean_and_deduplicate_data(input_jsonl, cleaned_output_jsonl, redundant_output_jsonl)

    print("\n--- Cleaning Process Complete ---")
    print(f"Non-redundant data saved to: {cleaned_output_jsonl}")
    print(f"Redundant (duplicate) data saved to: {redundant_output_jsonl}")
    print("\nNext Steps:")
    print("1. Please review the 'redundant_skills_data.jsonl' file. This file contains the records that were identified as duplicates and whose information was merged into the canonical records. You can update this file for your records or future use.")
    print("2. Remember to first manually fix any *circular dependencies* in your original 'all_skills.jsonl' file (identified by data_issues_reporter.py), as this script does not resolve them automatically.")
    print("3. Also, if you have *fuzzy duplicates* (e.g., 'Python' vs 'Python Programming'), ensure you've unified their names in the input file before running this cleaner. If you find any after running this, you'll need to manually update 'cleaned_skills_data.jsonl' if needed after inspection.")
    print("4. After making any further updates, you can use 'data/processed/cleaned_skills_data.jsonl' for further processing in your project.")
    print("5. Run `data_validator.py` on the `cleaned_skills_data.jsonl` file to re-validate and ensure all issues are resolved.")


if __name__ == '__main__':
    main()