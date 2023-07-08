import sys
from pathlib import Path
import csv
import os
import requests
# Read input data
CSV_FILENAME = str(sys.argv[1])
OUTPUT_FOLDER = str(sys.argv[2])
ID_INDEX = int(sys.argv[3])
URL_INDEX = int(sys.argv[4])
# Validate inputs
csv_path = Path(f"{CSV_FILENAME}.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"{csv_path} does not exist")
if (not isinstance(ID_INDEX, int)) or (not isinstance(URL_INDEX, int)):
    raise TypeError("indices must be integers")
if (ID_INDEX < 0) or (URL_INDEX < 0):
    raise ValueError("indices cannot be negative")
# Open CSV file
with open(csv_path, "r", encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    header_row = next(csv_reader)  # Skip the header row
    if ID_INDEX >= len(header_row):
        raise ValueError(
            f"ID_INDEX {ID_INDEX} is out of range for {CSV_FILENAME}.csv")
    if URL_INDEX >= len(header_row):
        raise ValueError(
            f"URL_INDEX {URL_INDEX} is out of range for {CSV_FILENAME}.csv")

    total_images = sum(1 for _ in csv_reader)
    csv_file.seek(0)  # Rewind the file pointer
    next(csv_reader)  # Skip the header row again
    # Initialize failure list and success counter
    success_counter = 0
    failures = []
    # Download images line by line
    for i, line in enumerate(csv_reader, start=1):
        image_id = line[ID_INDEX]
        image_url = line[URL_INDEX]
        image_path = Path(OUTPUT_FOLDER) / f"{image_id}.jpg"
        if image_path.exists():
            success_counter += 1
            print(f"Image {i} of {total_images} already exists")
        elif image_url:
            try:
                response = requests.get(image_url, stream=True, timeout=(20, 20))
                response.raise_for_status()
                # create parent directories if they don't exist
                os.makedirs(image_path.parent, exist_ok=True)
                with open(image_path, "wb") as image_file:
                    for chunk in response.iter_content(chunk_size=1024):
                        image_file.write(chunk)
                success_counter += 1
                print(f"Downloaded image {i} of {total_images}")
            except requests.exceptions.HTTPError:
                print(f"Failed to download {image_id} from {image_url}")
                failures.append(image_id)
            except requests.exceptions.Timeout:
                print(f"Timeout error while downloading {image_id} from {image_url}")
                failures.append(image_id)
            except requests.exceptions.ConnectionError:
                print(f"Connection error while downloading {image_id} from {image_url}")
                failures.append(image_id)
        else:
            print(f"No result for {image_id}")
            failures.append(image_id)
success_percentage = success_counter / total_images * 100
print(f"Downloaded {success_percentage}% of images successfully")
if failures:
    print(f"Failed to download images with IDs: {', '.join(failures)}")
