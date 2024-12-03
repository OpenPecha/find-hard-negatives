import requests
import json
from pathlib import Path
import os


def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Data successfully written to {file_path}")


def send_request_to_api(url, data):
    """Sends a POST request to the API with the provided data."""
    try:
        response = requests.post(url, json=data)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": "An error occurred while making the request", "details": str(e)}


def process_entries(data_list, api_url):
    responses = []
    for entry in data_list:
        if "image_url" in entry and "OCR_model" in entry:
            response = send_request_to_api(api_url, entry)
            responses.append(response)
        else:
            responses.append({
                "error": "Invalid entry",
                "details": "Missing 'image_url' or 'OCR_model'"
            })
    return responses


def process_directory(input_dir, output_dir, api_url):
    """Processes all JSON files in the input directory and saves responses in the output directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for input_file in input_path.glob("*.json"):
        data_list = read_json_file(input_file)
        if data_list is None:
            continue
        responses = process_entries(data_list, api_url)
        output_file = output_path / input_file.name
        write_json_file(responses, output_file)


if __name__ == "__main__":
    input_dir = "data/input_json"
    output_dir = "data/ocr_output"
    api_url = os.getenv("OCR_API_URL")
    process_directory(input_dir, output_dir, api_url)
