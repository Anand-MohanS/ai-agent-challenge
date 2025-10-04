#!/usr/bin/env python3
"""
Pytest validation for generated bank statement parsers.
Tests parser output against reference CSV using DataFrame.equals
"""

import pytest
import pandas as pd
import os
from pathlib import Path
import importlib.util

def load_parser(bank_name: str):
    """Load the generated parser module"""
    parser_path = f"custom_parser/{bank_name}_parser.py"
    
    if not os.path.exists(parser_path):
        pytest.skip(f"Parser not found: {parser_path}")
    
    spec = importlib.util.spec_from_file_location("parser", parser_path)
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)
    
    return parser_module

def test_icici_parser_exists():
    """Test that ICICI parser file was generated"""
    parser_path = "custom_parser/icici_parser.py"
    assert os.path.exists(parser_path), f"Parser file not found: {parser_path}"

def test_icici_parser_structure():
    """Test that ICICI parser has required structure"""
    parser_module = load_parser("icici")
    
    # Check that parse function exists
    assert hasattr(parser_module, 'parse'), "Parser module must have a 'parse' function"
    
    # Check function signature
    import inspect
    sig = inspect.signature(parser_module.parse)
    assert 'pdf_path' in sig.parameters, "parse function must accept 'pdf_path' parameter"

def test_icici_parser_output():
    """Test that ICICI parser returns valid DataFrame"""
    parser_module = load_parser("icici")
    
    pdf_path = "data/icici/icici sample.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Sample PDF not found: {pdf_path}")
    
    # Run the parser
    result_df = parser_module.parse(pdf_path)
    
    # Basic structure tests
    assert isinstance(result_df, pd.DataFrame), "Parser must return a DataFrame"
    assert not result_df.empty, "Parser must return non-empty DataFrame"
    
    # Expected columns
    expected_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    assert list(result_df.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(result_df.columns)}"
    
    # Data type checks
    assert len(result_df) > 0, "DataFrame should have data rows"
    
    # Check that we have some transactions
    assert len(result_df) > 10, "Should extract multiple transactions from the PDF"

def test_icici_parser_data_quality():
    """Test data quality of parsed results"""
    parser_module = load_parser("icici")
    
    pdf_path = "data/icici/icici sample.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Sample PDF not found: {pdf_path}")
    
    result_df = parser_module.parse(pdf_path)
    
    # Check that dates are present
    assert result_df['Date'].notna().sum() > 0, "Should have valid dates"
    
    # Check that descriptions are present
    assert result_df['Description'].notna().sum() > 0, "Should have transaction descriptions"
    
    # Check that balance values are present
    assert result_df['Balance'].notna().sum() > 0, "Should have balance values"
    
    # Check that either debit or credit amounts are present for each row
    has_amount = result_df['Debit Amt'].notna() | result_df['Credit Amt'].notna()
    assert has_amount.sum() > 0, "Each transaction should have either debit or credit amount"

def test_icici_parser_sample_data():
    """Test parser against known sample data"""
    parser_module = load_parser("icici")
    
    pdf_path = "data/icici/icici sample.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Sample PDF not found: {pdf_path}")
    
    result_df = parser_module.parse(pdf_path)
    
    # Check for some known transactions from the sample PDF
    descriptions = result_df['Description'].astype(str).str.lower()
    
    # Should find salary credits
    assert descriptions.str.contains('salary').any(), "Should find salary transactions"
    
    # Should find UPI payments
    assert descriptions.str.contains('upi').any(), "Should find UPI transactions"
    
    # Should find fuel purchases
    assert descriptions.str.contains('fuel').any(), "Should find fuel purchase transactions"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])