import sys
from pathlib import Path
import csv
import os
import requests

def image_from_csv_reader(csv_filename, output_folder):
    # Read input data
    csv_path = Path(f"{csv_filename}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} does not exist")

    # Open CSV file
    with open(csv_path, "r", encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        header_row = next(csv_reader)  # Skip the header row

        # Find the ID index as the column named "global_key"
        id_index = header_row.index("global_key")

        # Find the URL index as the column containing "_url" in the column name
        url_index = next((i for i, col in enumerate(header_row) if "_url" in col), None)
        if url_index is None:
            raise ValueError("No URL column found")

        total_images = sum(1 for _ in csv_reader)
        csv_file.seek(0)  # Rewind the file pointer
        next(csv_reader)  # Skip the header row again

        # Initialize failure list, success counter, and unavailable indices
        success_counter = 0
        failures = []
        unavailable_indices = []

        # Download images line by line
        for i, line in enumerate(csv_reader, start=1):
            image_id = line[id_index]
            image_url = line[url_index]
            image_path = Path(output_folder) / f"{image_id}.jpg"

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
                unavailable_indices.append(i)

        success_percentage = success_counter / total_images * 100
        print(f"Downloaded {success_percentage}% of images successfully")
        if failures:
            print(f"Failed to download images with IDs: {', '.join(failures)}")

        # Return the list of unavailable indices
        return unavailable_indices

csv_filename = sys.argv[1]
output_folder = sys.argv[2]

unavailable_indices = image_from_csv_reader(csv_filename, output_folder)

print(f"Unavailable indices: {unavailable_indices}")
