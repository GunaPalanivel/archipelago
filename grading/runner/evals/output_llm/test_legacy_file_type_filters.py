"""Legacy Studio file-type labels must keep filtering (not fall back to All output)."""

from runner.evals.output_llm.artifact_filters import (
    convert_file_types_to_extensions,
    is_valid_file_type,
)


def test_legacy_word_documents_label_is_accepted():
    legacy = "Word Documents (.docx, .doc)"
    assert is_valid_file_type(legacy)
    assert convert_file_types_to_extensions(legacy) == [".docx", ".doc"]


def test_current_word_documents_label_still_works():
    current = "Word Documents (.docx, .doc, .odt)"
    assert is_valid_file_type(current)
    assert convert_file_types_to_extensions(current) == [".docx", ".doc", ".odt"]


def test_legacy_spreadsheet_and_presentation_labels():
    assert convert_file_types_to_extensions("Spreadsheets (.xlsx, .xls, .xlsm)") == [
        ".xlsx",
        ".xls",
        ".xlsm",
    ]
    assert convert_file_types_to_extensions("Presentations (.pptx, .ppt)") == [
        ".pptx",
        ".ppt",
    ]
