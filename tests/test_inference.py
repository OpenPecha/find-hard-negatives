import pytest
from unittest.mock import patch, MagicMock
import json
import os
from pathlib import Path
from src.hard_negatives.inference import read_json_file, write_json_file, send_request_to_api, process_entries, process_directory


# Test for read_json_file function
def test_read_json_file():
    test_file = "test.json"
    test_data = {"key": "value"}
    with open(test_file, "w", encoding="utf-8") as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)

    result = read_json_file(test_file)
    assert result == test_data
    os.remove(test_file)


# Test for write_json_file function
def test_write_json_file(tmp_path):
    data = {"key": "value"}
    output_file = tmp_path / "output.json"

    write_json_file(data, output_file)

    with open(output_file, "r", encoding="utf-8") as file:
        result = json.load(file)

    assert result == data


# Test send_request_to_api function
@patch("requests.post")
def test_send_request_to_api(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "success"}
    mock_post.return_value = mock_response

    url = "http://example.com/api"
    data = {"key": "value"}

    response = send_request_to_api(url, data)

    assert response == {"status": "success"}
    mock_post.assert_called_once_with(url, json=data)


# Test process_entries function
def test_process_entries():
    test_data = [
        {"image_url": "http://example.com/image1", "OCR_model": "model1"},
        {"image_url": "http://example.com/image2", "OCR_model": "model2"},
        {"image_url": "http://example.com/image3"}
    ]
    api_url = "http://example.com/api"

    # Mock send_request_to_api
    with patch("src.hard_negatives.inference.send_request_to_api") as mock_send_request:
        mock_send_request.return_value = {"status": "success"}

        responses = process_entries(test_data, api_url)

        assert len(responses) == 3
        assert responses[0] == {"status": "success"}
        assert responses[1] == {"status": "success"}
        assert responses[2] == {"error": "Invalid entry", "details": "Missing 'image_url' or 'OCR_model'"}


# Test process_directory function
@patch("src.hard_negatives.inference.read_json_file")
@patch("src.hard_negatives.inference.write_json_file")
@patch("src.hard_negatives.inference.send_request_to_api")
def test_process_directory(mock_send_request, mock_write_json, mock_read_json, tmp_path):
    test_data = [
        {"image_url": "http://example.com/image1", "OCR_model": "model1"},
        {"image_url": "http://example.com/image2", "OCR_model": "model2"}
    ]
    input_dir = tmp_path / "input_json"
    output_dir = tmp_path / "ocr_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / "test_file.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    mock_read_json.return_value = test_data
    mock_send_request.return_value = {"status": "success"}
    process_directory(str(input_dir), str(output_dir), "http://example.com/api")

    mock_write_json.assert_called_once_with(
        [{"status": "success"}, {"status": "success"}], output_dir / "test_file.json"
    )
