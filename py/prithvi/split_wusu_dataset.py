"""
Script che divide il test set del dataset WUSU in set di validazione e test basati su aree di interesse (AoI).
Le aree di interesse del test set originale sono arregnate 50/50 a test e validazione. Lo script riproduce la 
struttura delle cartelle del dataset e sposta le immagini, SS mask, le BCD mask e le SCD mask rispettando la 
struttura originale.
"""

import os
import random
import shutil
import re
from pathlib import Path

# Imposta un seed per la riproducibilità
random.seed(42)

# Percorsi di origine e destinazione
SRC_BASE = "/shared/marangi/projects/EVOCITY/building_extraction/data/OpenWUSU512/test"
VAL_BASE = "/shared/marangi/projects/EVOCITY/building_extraction/data/WUSU_preprocessed/val"
TEST_BASE = "/shared/marangi/projects/EVOCITY/building_extraction/data/WUSU_preprocessed/test"

def create_directory_structure():
    """Crea la struttura delle directory necessaria."""
    for city in ['HS', 'JA']:
        for base_dir in [VAL_BASE, TEST_BASE]:
            os.makedirs(os.path.join(base_dir, city, 'imgs'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, city, 'class'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, city, 'change', 'BCD'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, city, 'change', 'SCD'), exist_ok=True)

def extract_aoi(filename):
    """Estrae l'Area of Interest (AoI) dal nome del file."""
    match = re.search(r'_(\d+)\.tif$', filename)
    if match:
        return match.group(1)
    return None

def get_aoi_split():
    """Divide le AoI di ogni città in due gruppi per val e test."""
    aois_by_city = {}
    
    for city in ['HS', 'JA']:
        img_path = os.path.join(SRC_BASE, city, 'imgs')
        all_files = os.listdir(img_path)
        
        # Estrai AoI uniche
        aois = set(extract_aoi(file) for file in all_files if extract_aoi(file))
        aois_list = list(aois)
        random.shuffle(aois_list)
        
        # Dividi in due gruppi
        split_point = len(aois_list) // 2
        val_aois = set(aois_list[:split_point])
        test_aois = set(aois_list[split_point:])
        
        print(f"Città {city}: {len(aois)} AoI totali, {len(val_aois)} per val, {len(test_aois)} per test")
        aois_by_city[city] = {'val': val_aois, 'test': test_aois}
    
    return aois_by_city

def copy_files_by_aoi(src_dir, val_dir, test_dir, city_aois):
    """Copia i file nelle cartelle appropriate in base all'AoI."""
    if not os.path.exists(src_dir):
        print(f"ATTENZIONE: Directory {src_dir} non trovata")
        return 0, 0
    
    val_count = test_count = 0
    
    for file in os.listdir(src_dir):
        aoi = extract_aoi(file)
        if not aoi:
            continue
            
        src_file = os.path.join(src_dir, file)
        
        if aoi in city_aois['val']:
            dest_file = os.path.join(val_dir, file)
            shutil.copy2(src_file, dest_file)
            val_count += 1
        else:
            dest_file = os.path.join(test_dir, file)
            shutil.copy2(src_file, dest_file)
            test_count += 1
            
    return val_count, test_count

def main():
    """Esegue la divisione del dataset."""
    print("Creazione della struttura delle directory...")
    create_directory_structure()
    
    print("Determinazione della divisione delle AoI...")
    aois_by_city = get_aoi_split()
    
    print("Copia dei file...")
    total_stats = {'val': 0, 'test': 0}
    
    for city in ['HS', 'JA']:
        city_stats = {'val': 0, 'test': 0}
        
        # Immagini e classi
        for subdir in ['imgs', 'class']:
            src_dir = os.path.join(SRC_BASE, city, subdir)
            val_dir = os.path.join(VAL_BASE, city, subdir)
            test_dir = os.path.join(TEST_BASE, city, subdir)
            
            val_count, test_count = copy_files_by_aoi(src_dir, val_dir, test_dir, aois_by_city[city])
            city_stats['val'] += val_count
            city_stats['test'] += test_count
            print(f"  {city}/{subdir}: {val_count} file in val, {test_count} file in test")
        
        # Change detection (BCD e SCD)
        for change_type in ['BCD', 'SCD']:
            src_dir = os.path.join(SRC_BASE, city, 'change', change_type)
            val_dir = os.path.join(VAL_BASE, city, 'change', change_type)
            test_dir = os.path.join(TEST_BASE, city, 'change', change_type)
            
            val_count, test_count = copy_files_by_aoi(src_dir, val_dir, test_dir, aois_by_city[city])
            city_stats['val'] += val_count
            city_stats['test'] += test_count
            print(f"  {city}/change/{change_type}: {val_count} file in val, {test_count} file in test")
        
        print(f"Totale per {city}: {city_stats['val']} file in val, {city_stats['test']} file in test")
        total_stats['val'] += city_stats['val']
        total_stats['test'] += city_stats['test']
    
    print(f"\nTOTALE: {total_stats['val']} file in validation, {total_stats['test']} file in test")
    print("Divisione completata con successo!")

if __name__ == "__main__":
    main()