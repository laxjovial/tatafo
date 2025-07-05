# utils/date_parser.py

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

def parse_date_to_yyyymmdd(date_string: str) -> Optional[str]:
    """
    Attempts to parse a date string into YYYY-MM-DD format.
    Supports several common formats.

    Args:
        date_string (str): The date string to parse.

    Returns:
        Optional[str]: The date in YYYY-MM-DD format, or None if parsing fails.
    """
    formats = [
        "%Y-%m-%d",          # 2025-07-05
        "%m/%d/%Y",          # 07/05/2025
        "%d-%m-%Y",          # 05-07-2025
        "%Y/%m/%d",          # 2025/07/05
        "%B %d, %Y",         # July 05, 2025
        "%b %d, %Y",         # Jul 05, 2025
        "%d %B %Y",          # 05 July 2025
        "%d %b %Y",          # 05 Jul 2025
        "%Y%m%d"             # 20250705
    ]

    for fmt in formats:
        try:
            dt_obj = datetime.strptime(date_string, fmt)
            return dt_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date string '{date_string}' into YYYY-MM-DD format.")
    return None

if __name__ == "__main__":
    print("--- Testing Date Parser ---")
    
    test_dates = [
        "2025-07-05",
        "07/05/2025",
        "5-7-2025",
        "2025/07/05",
        "July 05, 2025",
        "Jul 05, 2025",
        "05 July 2025",
        "05 Jul 2025",
        "20250705",
        "Invalid Date",
        "2025-13-01" # Invalid month
    ]

    for date_str in test_dates:
        parsed_date = parse_date_to_yyyymmdd(date_str)
        print(f"'{date_str}' -> '{parsed_date}'")
        if "Invalid" not in date_str and "13-01" not in date_str:
            assert parsed_date == "2025-07-05" or (date_str == "2025-07-05" and parsed_date == "2025-07-05"), f"Failed for {date_str}"
        elif "Invalid" in date_str or "13-01" in date_str:
            assert parsed_date is None, f"Should be None for invalid date: {date_str}"

    print("\nAll date parser tests completed.")
