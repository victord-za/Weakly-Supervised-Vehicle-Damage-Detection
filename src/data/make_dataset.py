import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import requests
import os

class MakeDataset:
    def __init__(self):
        self.df = None

        # Set up logging configuration
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)

        # Load environment variables from .env file
        load_dotenv(find_dotenv())

        # Read input data
        print('This is the csv file:' + os.getenv('CSV_FILENAME'))
        csv_filename = os.getenv('CSV_FILENAME')
        self.data_folder = os.getenv('DATA_FOLDER')
        self.image_folder = os.getenv('IMAGE_FOLDER')

        # Perform data processing steps
        unavailable_images = self.image_from_csv_reader(csv_filename)
        self.clean_data(unavailable_images)
        self.one_hot_encode_damage()
        self.save_to_csv()

    def get_csv(self, csv_filename):
        if isinstance(csv_filename, str):
            # Read CSV file if filename is provided
            csv_path = Path(csv_filename)
            if not csv_path.exists():
                raise FileNotFoundError(f"{csv_path} does not exist")
            self.df = pd.read_csv(csv_path)
        elif isinstance(csv_filename, pd.DataFrame):
            # Use DataFrame directly
            self.df = csv_filename
        else:
            raise TypeError("Invalid input type. Expected DataFrame or filename to DataFrame.")

    def image_from_csv_reader(self, csv_filename):

        # Read CSV file
        self.get_csv(csv_filename)

        header_row = self.df.columns.tolist()

        # Find the ID index as the column named "global_key"
        id_index = header_row.index("global_key")

        # Find the URL index as the column containing "_url" in the column name
        url_index = next((i for i, col in enumerate(header_row) if "_url" in col), None)
        if url_index is None:
            raise ValueError("No URL column found")

        total_images = len(self.df)

        # Initialize failure list, success counter, and unavailable indices
        success_counter = 0
        failures = []
        unavailable_indices = []

        # Download images row by row
        for i, row in self.df.iterrows():
            image_id = row[id_index]
            image_url = row[url_index]
            image_path = Path(self.image_folder) / f"{image_id}.jpg"

            if image_path.exists():
                success_counter += 1
                print(f"Image {i+1} of {total_images} already exists")
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
                    print(f"Downloaded image {i+1} of {total_images}")
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

        self.available_image_percentage = success_counter / total_images * 100
        print(f"Downloaded {self.available_image_percentage}% of images successfully")
        if failures:
            print(f"Failed to download images with IDs: {', '.join(failures)}")

        # Return the list of unavailable indices
        return unavailable_indices
    
    def clean_data(self, unavailable_images):
        # Delete rows where images are not available
        self.df = self.df.drop(unavailable_images, axis=0).reset_index(drop=True)

        # Fill NaN values in damage_detail column with empty string
        self.df['damage_detail'].fillna('', inplace=True)

        # Drop columns that are not needed
        self.df.drop(['img_url'], axis=1, inplace=True)
        self.df.drop(['evaluation_type'], axis=1, inplace=True)

    def one_hot_encode_damage(self):
        one_hot_encoded = self.df['damage_detail'].str.get_dummies(sep=',')
        one_hot_encoded.columns = ['DAMAGE_' + col for col in one_hot_encoded.columns]

        self.df = pd.concat([self.df, one_hot_encoded], axis=1)
        self.df.drop(['damage_detail'], axis=1, inplace=True)

    def save_to_csv(self):
        output_filepath = Path(self.data_folder) / 'processed' / 'processed.csv'
        self.df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    dataset = MakeDataset()
