import pytest
from etl import parse_age_range, parse_numeric_years, parse_hours_per_week, skill_to_numeric, yes_no_to_bool

def test_parse_age_range_dash():
    assert parse_age_range("25-34") == ("25-34", 25, 34)

def test_parse_age_range_plus():
    assert parse_age_range("65+") == ("65+", 65, None)

def test_parse_years():
    assert parse_numeric_years("3 years") == 3.0
    assert parse_numeric_years("less than 1") == 0.5

def test_parse_hours():
    assert parse_hours_per_week("10") == 10.0
    assert parse_hours_per_week("10-15") == 12.5

def test_skill_mapping():
    assert skill_to_numeric("beginner") == 1
    assert skill_to_numeric("advanced") == 8

def test_yes_no_bool():
    assert yes_no_to_bool("yes") is True
    assert yes_no_to_bool("No") is False
    assert yes_no_to_bool("") is None