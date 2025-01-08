# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:08:32 2024

@author: danie

Processing the mid, bid and ask spreads from file Capital Buffers - Sample_v01.xlsx

"""
import os
import pandas as pd

def process_excel_to_csv(excel_path, output_folders):
    # Unpack output folders
    output_folder_mid, output_folder_bid, output_folder_ask = output_folders[0], output_folders[1], output_folders[2]
    # Ensure the output folders exist
    os.makedirs(output_folder_mid, exist_ok=True)
    os.makedirs(output_folder_bid, exist_ok=True)
    os.makedirs(output_folder_ask, exist_ok=True)

    # Load the Excel file
    xls = pd.ExcelFile(excel_path)

    # Loop through all sheets
    for sheet_name in xls.sheet_names:
        # Read data from columns A to F
        df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols='A:F')

        if df.empty or len(df.columns) < 2:
            print(f"Skipping sheet '{sheet_name}' due to insufficient data.")
            continue

        # Copy B1 to B2 and delete the first row
        if len(df) > 0:
            csv_name = df.columns[1]
            df.iloc[0, 1] = csv_name  # Copy B1 to B2
            df.iloc[0, 3] = csv_name  # Copy B1 to B2
            df.iloc[0, 5] = csv_name  # Copy B1 to B2

        # Remove rows that are completely empty
        df.dropna(how='all', inplace=True)

        # Ensure no empty or malformed `csv_name`
        if not isinstance(csv_name, str) or not csv_name.strip():
            print(f"Skipping sheet '{sheet_name}' because B1 is empty or invalid.")
            continue
        
        # Remove rows where any of the first 6 columns are completely empty
        df.dropna(subset=df.columns[:6], how='all', inplace=True)

        # Mid
        csv_path_mid = os.path.join(output_folder_mid, f"{csv_name.strip()}.csv")
        df.iloc[:, 0:2].dropna(how='all').to_csv(csv_path_mid, index=False, header=False)
        
        # Bid
        csv_path_bid = os.path.join(output_folder_bid, f"{csv_name.strip()}.csv")
        df.iloc[:, 2:4].dropna(how='all').to_csv(csv_path_bid, index=False, header=False)
        
        # Ask
        csv_path_ask = os.path.join(output_folder_ask, f"{csv_name.strip()}.csv")
        df.iloc[:, 4:6].dropna(how='all').to_csv(csv_path_ask, index=False, header=False)

        print(f"Processed and saved sheet '{sheet_name}' to '{csv_path_mid}', '{csv_path_bid}', and '{csv_path_ask}'.")

# Example usage
excel_path = r'data\Raw Data\Capital Buffers - Sample.xlsx'
output_folders = [r'data\cds_mid', r'data_v02\cds_bid', r'data_v02\cds_ask']

process_excel_to_csv(excel_path, output_folders)
