#!/usr/bin/env python3
import os
import glob
import csv
import argparse

def merge_csv_files(output_name):
    # Chemin vers le dossier baseline/results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    
    # Pattern de recherche
    pattern = os.path.join(results_dir, "results_*.csv")
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print(f"Aucun fichier correspondant à {pattern} n'a été trouvé.")
        return
    
    output_file = os.path.join(results_dir, output_name)
    
    print(f"Trouvé {len(csv_files)} fichiers à fusionner :")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
        
    all_rows = []
    all_fields = []
    
    # Détecter toutes les colonnes uniques dans l'ordre d'apparition
    for file_path in csv_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in all_fields:
                        all_fields.append(field)
                for row in reader:
                    all_rows.append(row)
                    
    # Écriture dans le fichier final
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(all_rows)
        
    print(f"\nFusion réussie ! {len(all_rows)} lignes écrites dans :")
    print(f"  -> {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusionne les fichiers CSV de résultats.")
    parser.add_argument(
        "-o", "--output", 
        default="merged_results.csv", 
        help="Nom du fichier de sortie (sauvegardé dans baseline/results/). Par défaut: merged_results.csv"
    )
    args = parser.parse_args()
    merge_csv_files(args.output)
