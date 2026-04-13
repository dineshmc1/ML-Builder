"""
delete_memory.py
================
A self-contained CLI utility to manage and delete the persisted FAISS meta-learning memory.

By default, the unified memory is persisted to disk at `models/unified_memory.json`.
The FAISS index itself is dynamically built from this file. So, to delete the memory 
(fully or partially), this script will remove the required entries from the persisted JSON.

Usage:
  python delete_memory.py
"""

import os
import json
import sys

MEMORY_PATH = os.path.join("models", "unified_memory.json")

def load_memory():
    """Load JSON records from memory path."""
    if not os.path.exists(MEMORY_PATH):
        return []
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("Error: The memory file is corrupted or not valid JSON.")
        return []

def save_memory(data):
    """Save the filtered data back to memory path."""
    os.makedirs(os.path.dirname(MEMORY_PATH) or ".", exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Memory saved successfully. Current records remaining: {len(data)}")

def print_menu():
    print("\n" + "="*50)
    print("      FAISS MEMORY MANAGEMENT MENU")
    print("="*50)
    print("1. View Current Memory Stats")
    print("2. Delete Entire Memory (Clear All)")
    print("3. Partially Delete by Dataset ID (e.g., openml_100)")
    print("4. Partially Delete by Paradigm (ML or DL)")
    print("5. Exit")
    print("="*50)

def main():
    if not os.path.exists(MEMORY_PATH):
        print(f"\nMemory file '{MEMORY_PATH}' does not exist.")
        print("Nothing to delete. You can pre-seed it by running `python preseed_memory.py`.")
        return

    data = load_memory()
    if not data:
        print("\nThe memory file is currently empty.")
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            print(f"\n--- Memory Stats ---")
            print(f"Total entries: {len(data)}")
            if len(data) > 0:
                datasets = sorted(list(set(d.get("dataset_id", "unknown") for d in data)))
                paradigms = sorted(list(set(d.get("paradigm", "unknown") for d in data)))
                print(f"Unique Datasets ({len(datasets)}): {datasets[:10]}{'...' if len(datasets) > 10 else ''}")
                print(f"Paradigms Present: {paradigms}")
            else:
                print("Memory is empty.")
            
        elif choice == '2':
            confirm = input("Are you sure you want to delete the ENTIRE memory? (y/n): ").strip().lower()
            if confirm == 'y':
                data = []
                save_memory(data)
                print("Entire memory cleared.")
            else:
                print("Operation cancelled.")
                
        elif choice == '3':
            datasets = sorted(list(set(d.get("dataset_id", "unknown") for d in data)))
            if not datasets:
                print("No datasets to delete.")
                continue
            print(f"Available Dataset IDs (Showing up to 15): {datasets[:15]}\n...")
            ds_id = input("Enter the Dataset ID to delete (or leave blank to cancel): ").strip()
            if not ds_id:
                continue
            
            new_data = [d for d in data if d.get("dataset_id") != ds_id]
            removed = len(data) - len(new_data)
            if removed > 0:
                print(f"Removed {removed} entries associated with dataset ID '{ds_id}'.")
                data = new_data
                save_memory(data)
            else:
                print(f"No entries found with dataset ID '{ds_id}'.")
                
        elif choice == '4':
            paradigm = input("Enter paradigm to delete (ML or DL) (or blank to cancel): ").strip().upper()
            if not paradigm:
                continue
                
            new_data = [d for d in data if d.get("paradigm") != paradigm]
            removed = len(data) - len(new_data)
            if removed > 0:
                print(f"Removed {removed} entries associated with paradigm '{paradigm}'.")
                data = new_data
                save_memory(data)
            else:
                print(f"No entries found with paradigm '{paradigm}'.")
                
        elif choice == '5':
            print("Exiting tool.")
            break
            
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting tool.")
        sys.exit(0)
