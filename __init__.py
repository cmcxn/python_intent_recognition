"""
Python Intent Recognition Module

A complete implementation of RoBERTa-based intent recognition for office domain tasks.
"""

__version__ = "1.0.0"
__author__ = "Intent Recognition System"

# Intent labels for office domain
INTENT_LABELS = [
    "salary_inquiry",
    "meeting_room_booking", 
    "leave_request",
    "directory_search",
    "company_info",
    "employee_info",
    "employee_search"
]

# Label to ID mapping
LABEL_TO_ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(INTENT_LABELS)}