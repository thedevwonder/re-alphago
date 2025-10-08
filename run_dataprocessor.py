#!/usr/bin/env python3
"""
Script to run the DataProcessor on SGF files in the data directory.
This script processes Go game data from SGF files and converts them into
training data for machine learning models.
"""

from asyncio import as_completed
import os
from site import execusercustomize
import sys
from concurrent.futures import ThreadPoolExecutor

# Add the current directory to Python path to import dlgo modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlgo.dataprocessor import DataProcessor


def main():
    """Main function to run the data processor."""
    
    # Configuration
    data_dir = "data"
    encoder_name = "fourplane"  # You can change this to other encoders if available
    file_list = []
    zip_files = []

    # get all sgf files in the dir
    for file in os.listdir(data_dir):
        if file.endswith('.sgf'):
            file_list.append(f"{data_dir}/{file}")
        if file.endswith('.tar.gz'):
            zip_files.append(f"{file}")
    
    print(f"Starting data processing...")
    print(f"Data directory: {data_dir}")
    print(f"Encoder: {encoder_name}")
    print(f"Total sgf files in dir: {len(file_list)}")
    print("-" * 50)
    
    try:
        # Initialize the data processor
        processor = DataProcessor(encoder_name, data_dir)
        
        # Process the SGF files
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = executor.map(lambda file: processor.process_sgf_files(zip_file_name=file), zip_files)

            for future in as_completed(futures):
                future.result()
    
        print("Data processing completed successfully!")
        print("Output files have been saved to the data directory.")
            
    except Exception as e:
        print(f"Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nData processing script completed!")


if __name__ == "__main__":
    main() 