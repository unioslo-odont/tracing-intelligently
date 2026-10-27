"""
================================================================================
JIAHONGQIAN (KAGGLE) DATASET CONVERTER
================================================================================
DESCRIPTION:
    One-off converter for the public "jiahongqian cephalometric landmarks"
    Kaggle dataset (not the Norwegian TSD data - this dataset has its own
    directory layout and its own fixed landmark set). For every .fcd found
    under base_folder:
      1. Parses patient sex/age from the .fcd.
      2. Converts the linked image (DICOM or standard format) to an
         anonymized 16-bit grayscale TIFF named after the .fcd file.
      3. Extracts the first occurrence of each landmark in LANDMARKS (in
         get_marker_coordinates below) by matching <marker> and <coord>
         elements positionally - this assumes markers and coords appear in
         the same relative order, which holds for this dataset's export
         format but is not a general .fcd parsing strategy (contrast with
         select_reviewed_tracings.py, which pairs <name>/<coord> within the
         same <marker> element instead).
      4. Writes one row per .fcd to facad_data.csv.
================================================================================
"""

import os
import xml.etree.ElementTree as ET
import csv
from datetime import datetime
import pydicom
from PIL import Image
import numpy as np
import shutil

base_folder = r"N:\durable\Tracing-intelligently\data\jiahongqian-cephalometric-landmarks-kaggle\Tracet"
anon_path = r"N:\durable\Tracing-intelligently\data\jiahongqian-cephalometric-landmarks-kaggle\anonymized"
output_csv = os.path.join(anon_path, 'facad_data.csv')
data = []

def convert_dicom_to_anon_tiff(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array
    if pixel_array.dtype != np.uint16:
        if pixel_array.max() <= 255:
            pixel_array = (pixel_array * 256).astype(np.uint16)
        else:
            pixel_array = pixel_array.astype(np.uint16)
    img = Image.fromarray(pixel_array, mode='I;16')
    img.save(output_path, format='TIFF')

def get_marker_coordinates(root, marker_name):
    markers = root.findall('.//marker')
    coords = root.findall('.//coord')
    for i, marker in enumerate(markers):
        if marker.findtext('name', '').strip() == marker_name and i < len(coords):
            coord_text = coords[i].text.strip()
            if ' ' in coord_text:
                return coord_text.split()
    return ('', '')

for dirpath, _, filenames in os.walk(base_folder):
    for filename in filenames:
        if filename.endswith('.fcd'):
            fcd_path = os.path.join(dirpath, filename)
            try:
                tree = ET.parse(fcd_path)
                root = tree.getroot()

                patient = root.find('patient')
 
                date_str = patient.findtext('time', '').strip().split()[0]
                birthdate_str = patient.findtext('birthdate', '').strip()
                age_years = 'NA'
                try:
                    if date_str and birthdate_str:
                        exam_date = datetime.strptime(date_str, '%Y-%m-%d')
                        birth_date = datetime.strptime(birthdate_str, '%Y-%m-%d')
                        age_years = round((exam_date - birth_date).days / 365.25)
                except ValueError:
                    pass  # Keep age_years = 'NA'
 
                sex = patient.findtext('sex', '').strip().lower()

                image_rel_path = root.findtext('.//imagefile/name', '').strip()
                image_filename = os.path.basename(image_rel_path)
                image_folder = os.path.join(dirpath, 'images')
                image_path = os.path.join(image_folder, image_filename)

                if os.path.exists(image_path):
                    dst_path = os.path.join(anon_path, os.path.splitext(filename)[0] + '.tiff')
                    if image_path.lower().endswith('.dcm'):
                        convert_dicom_to_anon_tiff(image_path, dst_path)
                    else:
                        img = Image.open(image_path)
                        img.convert('L').save(dst_path, format='TIFF')

                coords = {k: get_marker_coordinates(root, k) for k in [
                    'A', 'B', 'S', 'N', 'Or', 'Ba', 'Po', 'GL', 'PRN', 'MS', 'RN',
                    'PLs', 'STs', 'STi', 'PLi', 'LLV', 'PGs', 'GNs'
                ]}
                flat_coords = {f"{k}_x": x for k, (x, y) in coords.items()}
                flat_coords.update({f"{k}_y": y for k, (x, y) in coords.items()})

                data.append({
                    'imagefile': os.path.basename(dst_path),
                    'age': age_years,
                    'sex': sex,
                    **flat_coords
                })

            except ET.ParseError as e:
                print(f"Parse error in {fcd_path}: {e}")
            except Exception as e:
                print(f"Failed processing {fcd_path}: {e}")

if data:
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

print(f"Exported {len(data)} records to {output_csv}")
