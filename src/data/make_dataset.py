import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import requests
import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

class MakeDataset:
    def __init__(self):
        self.df = None

        # Set up logging configuration
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)

        # Load environment variables from .env file
        load_dotenv(find_dotenv())

        # Read input data
        csv_filename = os.getenv('CSV_FILENAME')
        self.data_folder = os.getenv('DATA_FOLDER')
        self.image_folder = os.getenv('IMAGE_FOLDER')

        # Perform data processing steps
        unavailable_images = self.image_from_csv_reader(csv_filename)
        self.clean_data(unavailable_images)
        self.one_hot_encode_damage()
        self.save_interim_data()
        #self.split_dataset()

    def get_csv(self, csv_filename):
        if isinstance(csv_filename, str):
            # Read CSV file if filename is provided
            csv_path = Path(csv_filename)
            if not csv_path.exists():
                raise FileNotFoundError(f"{csv_path} does not exist")
            self.df = pd.read_csv(csv_path, delimiter=",")
        elif isinstance(csv_filename, pd.DataFrame):
            # Use DataFrame directly
            self.df = csv_filename
        else:
            raise TypeError("Invalid input type. Expected DataFrame or filename to DataFrame.")

    def image_from_csv_reader(self, csv_filename):

        # Read CSV file
        self.get_csv(csv_filename)
        
        self.image_locations = np.zeros((self.df.shape[0], 3))

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

        print(f"Downloading {total_images} images...")
        # Download images row by row
        for i, row in self.df.iterrows():
            image_id = row[id_index]
            image_url = row[url_index]
            image_paths = [
                Path(self.image_folder) / f"{image_id}.jpg",
                Path(self.data_folder) / 'interim' /  f"{image_id}.jpg",
                Path(self.data_folder) / 'processed' / 'train' / f"{image_id}.jpg",
                Path(self.data_folder) / 'processed' / 'test' / f"{image_id}.jpg",
                Path(self.data_folder) / 'processed' / 'val' / f"{image_id}.jpg"
            ]
            self.image_locations[i, 0] = image_paths[0].exists()
            self.image_locations[i, 1] = image_paths[1].exists()
            self.image_locations[i, 2] = any(image_path.exists() for image_path in image_paths[2:])
            if any(self.image_locations[i, :]): #any(image_path.exists() for image_path in image_paths):
                success_counter += 1
            elif image_url:
                try:
                    response = requests.get(image_url, stream=True, timeout=(20, 20))
                    response.raise_for_status()
                    # create parent directories if they don't exist
                    image_path = image_paths[0]
                    os.makedirs(image_path.parent, exist_ok=True)
                    with open(image_path, "wb") as image_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            image_file.write(chunk)
                    success_counter += 1
                    self.image_locations[i, 0] = 1
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

        if failures:
            print(f"\nFailed to download images with IDs: {', '.join(failures)}")
        
        self.available_image_percentage = success_counter / total_images * 100
        print(f"\nDownloaded {self.available_image_percentage}% of images successfully")
        
        return unavailable_indices
    
    def clean_data(self, unavailable_images):
        # Delete rows where images are not available
        self.df = self.df.drop(unavailable_images, axis=0).reset_index(drop=True)
        self.image_locations = np.delete(self.image_locations, unavailable_images, axis=0)

        # Fill NaN values in damage_detail column with empty string
        self.df['damage_detail'].fillna('', inplace=True)

        # Drop columns that are not needed
        self.df.drop(['img_url'], axis=1, inplace=True)
        self.df.drop(['evaluation_type'], axis=1, inplace=True)
        
        # Remove rows of ExteriorPhotos_Front and ExteriorPhotos_Rear. These images are taken from afar and are labeled with all the underlying damaged components.
        mask = ~self.df['component'].isin(['ExteriorPhotos_Front', 'ExteriorPhotos_Rear'])
        self.df = self.df[mask].reset_index(drop=True)
        self.image_locations[mask]
        

    def one_hot_encode_damage(self):
        if any(self.image_locations[:, 0]):
            # One hot encode damage detail column
            one_hot_encoded = self.df['damage_detail'].str.get_dummies(sep=',')
            one_hot_encoded.columns = ['DAMAGE_' + col for col in one_hot_encoded.columns]
            one_hot_encoded = one_hot_encoded.rename(columns={"DAMAGE_BROKEN-OFF_COMPONENTS": "DAMAGE_MISSING"})
            one_hot_encoded = one_hot_encoded.drop('DAMAGE_AS_EXPECTED', axis=1)
            
            # One hot encode condition and quality columns
            one_hot_encoded['DAMAGE_CRACKS'] = ((self.df['condition'] == 'CRACKS') | 
                                (self.df['condition'] == 'CRACKED')).astype(int)
            one_hot_encoded['DAMAGE_CHIPS'] = (self.df['condition'] == 'CHIPS').astype(int)
            one_hot_encoded['DAMAGE_PAINT'] = ((self.df['condition'] == 'FADED') | 
                            (self.df['condition'] == 'NEEDS_POLISH') | 
                            (self.df['quality'] == 'BELOW_AVERAGE')).astype(int)
            one_hot_encoded['DAMAGE_MISSING'] = ((self.df['condition'] == 'MISSING') | 
                                (one_hot_encoded['DAMAGE_MISSING'] == 1)).astype(int)
            temp_damage = ((self.df['condition'] == 'DAMAGE') | 
                                (self.df['condition'] == 'MINOR_DAMAGE') | 
                                (self.df['condition'] == 'MAJOR_DAMAGE')).astype(int)
            one_hot_encoded['DAMAGE_MISC'] = (temp_damage & one_hot_encoded.sum(axis=1) == 0)

            self.df = pd.concat([self.df, one_hot_encoded], axis=1)
            self.df.drop(['damage_detail', 'condition', 'quality', 'quality_detail'], axis=1, inplace=True)
        

    def save_interim_data(self):
        #output_folderpath = Path(self.data_folder) / 'interim'
        #output_filepath = os.path.join(output_folderpath, 'interim.csv')
        #self.df.to_csv(output_filepath, index=False)
        #header_row = self.df.columns.tolist()
        # Find the ID index as the column named "global_key"
        #id_index = header_row.index("global_key")
        #for i, row in self.df.iterrows():
        #    image_id = row[id_index]
        #    image_path = Path(self.image_folder) / f"{image_id}.jpg"
        #    if image_path.exists():
        #        shutil.move(image_path, output_folderpath)
        output_folderpath = Path(self.data_folder) / 'interim'
        output_filepath = os.path.join(output_folderpath, 'interim.csv')
        if any(self.image_locations[:, 0]):
            self.df.to_csv(output_filepath, index=False)

            def move_image(row):
                image_id = row["global_key"]
                image_path = Path(self.image_folder) / f"{image_id}.jpg"
                if image_path.exists():
                    shutil.move(image_path, output_folderpath)

            self.df.apply(move_image, axis=1)
        else:
            self.df = pd.read_csv(output_filepath, delimiter=',')
                

if __name__ == '__main__':
    dataset = MakeDataset()
