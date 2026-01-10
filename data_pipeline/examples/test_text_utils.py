#!/usr/bin/env python3
"""
Test the text utilities.
"""
import sys
sys.path.append('.')

from utilities.text_utils import (
    clean_article_text,
    extract_title,
    extract_location,
    extract_actors
)

# Example usage
raw_text = """
Israeli forces raided the town of Hebron today, arresting several Palestinians.
The incident occurred in the Old City area.
HEBRON, Monday, January 15, 2024 (WAFA) - Israeli forces conducted a raid in 
the Old City of Hebron, arresting five Palestinian youths. 
The soldiers also confiscated several vehicles.

Share this article on social media.
Follow WAFA on Twitter for more updates.
"""

print("=" * 60)
print("TESTING TEXT UTILITIES")
print("=" * 60)

# Clean the text
clean_text = clean_article_text(raw_text)
print(f"\n1. CLEANED TEXT:")
print(f"Original length: {len(raw_text)} chars")
print(f"Cleaned length: {len(clean_text)} chars")
print(f"\nFirst 200 chars of cleaned text:\n{clean_text[:200]}...")

# Extract title
title = extract_title(clean_text)
print(f"\n2. EXTRACTED TITLE:\n{title}")

# Extract location
location = extract_location(clean_text)
print(f"\n3. EXTRACTED LOCATION:\n{location}")

# Extract actors
actors = extract_actors(clean_text)
print(f"\n4. EXTRACTED ACTORS:\n{actors}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)