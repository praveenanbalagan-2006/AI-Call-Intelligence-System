import os
import json
from typing import Dict, Any

LOCAL_RECORDS_FILE = "analysis_records.json"

def save_analysis_to_firestore(app_id: str, record: Dict[str, Any]) -> bool:
    """
    Saves the structured call analysis record to a local JSON file.
    :param app_id: The application ID (ignored for local storage).
    :param record: The complete structured analysis data.
    :return: True if saved successfully, False otherwise.
    """
    try:
        if os.path.exists(LOCAL_RECORDS_FILE):
            with open(LOCAL_RECORDS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            records = []
        records.append(record)
        with open(LOCAL_RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"💾 Analysis saved to local file: {LOCAL_RECORDS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving record to local file: {e}")
        return False

def fetch_all_analysis_records(app_id: str):
    """
    Fetches all analysis records from the local JSON file.
    Returns a list of records (dicts).
    """
    try:
        if os.path.exists(LOCAL_RECORDS_FILE):
            with open(LOCAL_RECORDS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
            return records
        else:
            return []
    except Exception as e:
        print(f"Error fetching records from local file: {e}")
        return []