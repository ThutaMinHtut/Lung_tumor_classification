import os
import csv
import pydicom
import numpy as np
import cv2
from tqdm import tqdm

#SAMPLE IMAGE OUTPUT NAME :1.3.6.1.4.1.14519.5.2.1.6655.2359.100034712436708328075639143222_1.3.6.1.4.1.14519.5.2.1.6655.2359.101970556727310889524215117803

# Paths
dicom_root = "PETCTDX/manifest/"  # Only point to "manifest", not the full DICOM path
metadata_file = "PETCTDX/manifest/metadata.csv"  # Metadata CSV file
output_root = "ExtractedCT_UID512/"  # Output directory

# Define tumor classes based on patient ID
tumor_classes = {
    "A": "Adenocarcinoma",
    "E": "Large_Cell_Carcinoma",
    "G": "Squamous_Cell_Carcinoma"
}

# Ensure output directories exist
for class_name in tumor_classes.values():
    os.makedirs(os.path.join(output_root, class_name), exist_ok=True)

# Function to normalize and convert DICOM to PNG format (512x512)
def dicom_to_png(dicom_data):
    """Normalize pixel values and convert DICOM slice to 512x512 PNG."""
    image = dicom_data.pixel_array.astype(np.float32)

    # Normalize to 0-255
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image * 255).astype(np.uint8)

    # Resize to 512x512
    image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    return image_resized

# Read metadata.csv and collect CT series information
ct_series = []
with open(metadata_file, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row["Modality"] == "CT":  # Ensure it's a CT scan, not PET
            subject_id = row["Subject ID"]
            tumor_type = None
            for key in tumor_classes:
                if key in subject_id:
                    tumor_type = tumor_classes[key]
                    break
            
            if tumor_type:
                # Fix duplicate path issue
                series_path = os.path.normpath(os.path.join(dicom_root, row["File Location"].lstrip('.\\')))
                ct_series.append((tumor_type, series_path, row["Series UID"]))

# Check if any CT series were found
if not ct_series:
    print(" No CT series found in metadata.csv! Check file paths.")
    exit()

# Process each CT series
for tumor_type, series_path, series_uid in tqdm(ct_series, desc="Processing CT Scans"):
    print(f"\n Checking series: {series_uid} (Tumor: {tumor_type})")
    
    if not os.path.exists(series_path):
        print(f" Series path does not exist: {series_path}")
        continue

    dicom_slices = []
    dicom_files = []

    # Loop through series folders to find DICOM files
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith(".dcm"):  # Process only DICOM files
                dicom_file_path = os.path.join(root, file)
                try:
                    dicom_data = pydicom.dcmread(dicom_file_path)
                    dicom_slices.append(dicom_data)
                    dicom_files.append(dicom_file_path)
                except Exception as e:
                    print(f" Error reading DICOM file {dicom_file_path}: {e}")

    # If no DICOM slices found, print a warning
    if not dicom_slices:
        print(f" No DICOM files found in {series_path}")
        continue

    # Sort slices by Instance Number
    try:
        dicom_slices.sort(key=lambda x: int(x.InstanceNumber))
        dicom_files.sort()
    except AttributeError:
        print(f" Error: Some DICOM files in {series_path} are missing 'InstanceNumber'!")
        continue

    # Determine which slices to extract
    total_slices = len(dicom_slices)
    
    if tumor_type == "Large_Cell_Carcinoma":
        selected_slices = dicom_slices  # Take all slices
        selected_files = dicom_files
        print(f" Extracting ALL {total_slices} slices for {tumor_type}")
    else:
        # Extract only 45%-65% of slices for other tumor types
        start_idx = int(0.45 * total_slices)
        end_idx = int(0.65 * total_slices)
        selected_slices = dicom_slices[start_idx:end_idx]
        selected_files = dicom_files[start_idx:end_idx]
        print(f" Extracting {len(selected_slices)} slices from {total_slices} for {tumor_type}")

    # Convert and save only selected slices
    for dicom_data, dicom_file_path in zip(selected_slices, selected_files):
        try:
            image = dicom_to_png(dicom_data)

            # Extract DICOM UID
            dicom_uid = getattr(dicom_data, "SOPInstanceUID", None)
            if dicom_uid is None:
                print(f" Missing SOPInstanceUID for {dicom_file_path}, skipping...")
                continue

            file_name = f"{series_uid}_{dicom_uid}.png"
            save_path = os.path.join(output_root, tumor_type, file_name)

            cv2.imwrite(save_path, image)
            print(f" Saved: {save_path}")
        except Exception as e:
            print(f" Error processing {dicom_file_path}: {e}")

print("\n DICOM Extraction & Conversion Complete")