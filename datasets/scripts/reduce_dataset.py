import os
import argparse

def load_names(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def remove_files_with_patterns(root_folder, names):
    removed_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            for name in names:
                if filename == f"{name}.png" or filename == f"{name}.txt":
                    file_path = os.path.join(dirpath, filename)
                    try:
                        os.remove(file_path)
                        removed_files.append(file_path)
                    except Exception as e:
                        print(f"Failed to remove {file_path}: {e}")
                    break
    print(f"Total files removed: {len(removed_files)}")
    print("Removed files:", removed_files)

def main():
    parser = argparse.ArgumentParser(description="Remove .png and .txt files matching base names listed in a txt file.")
    parser.add_argument('--folder', type=str, required=True, help='Path to the root folder')
    parser.add_argument('--remove-list', type=str, required=True, help='Path to the txt file with base names (without extension)')
    args = parser.parse_args()

    names = load_names(args.remove_list)
    remove_files_with_patterns(args.folder, names)

if __name__ == "__main__":
    main()