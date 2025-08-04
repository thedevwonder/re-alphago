#!/usr/bin/env python3
"""
Script to run the DataProcessor on SGF files in the data directory.
This script processes Go game data from SGF files and converts them into
training data for machine learning models.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path to import dlgo modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlgo.dataprocessor.dataprocessor import DataProcessor


def main():
    """Main function to run the data processor."""
    
    # Configuration
    data_dir = "data"
    encoder_name = "oneplane"  # You can change this to other encoders if available
    zip_file_name = "KGS-2019_04-19-1255-.tar.gz"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please make sure the data directory exists and contains the SGF files.")
        return
    
    # Check if the zip file exists
    zip_file_path = os.path.join(data_dir, zip_file_name)
    if not os.path.exists(zip_file_path):
        print(f"Error: Zip file '{zip_file_path}' not found!")
        print("Please make sure the zip file exists in the data directory.")
        return
    
    print(f"Starting data processing...")
    print(f"Data directory: {data_dir}")
    print(f"Encoder: {encoder_name}")
    print(f"Input file: {zip_file_name}")
    print("-" * 50)
    
    try:
        # Initialize the data processor
        processor = DataProcessor(encoder_name, data_dir)
        
        # Process the SGF files
        processor.process_sgf_files(zip_file_name)
        
        print("Data processing completed successfully!")
        print("Output files have been saved to the data directory.")
        
        # List the generated files
        print("\nGenerated files:")
        base_name = zip_file_name.replace('.tar.gz', '')
        data_file_name = os.path.join(data_dir, base_name + '_train')
        
        # Check for generated files
        generated_files = []
        for file in os.listdir(data_dir):
            if file.startswith(base_name + '_train_') and file.endswith('.npy'):
                generated_files.append(file)
        
        if generated_files:
            for file in sorted(generated_files):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  {file} ({file_size:,} bytes)")
        else:
            print("  No output files found. Check for errors in processing.")
            
    except Exception as e:
        print(f"Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nData processing script completed!")


if __name__ == "__main__":
    main() 