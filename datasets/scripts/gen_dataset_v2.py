import argparse
import ast
import os
import shutil
from tqdm import tqdm

### USAGE: python gen_dataset_v2.py --dictionary '{"D1": 100, "D2": 150}' --new-dataset CombinedDataset --split-ratio "[70,15,15]"
# --dictionary '{"HRSID_YOLO_png_640px": 5000, "SSDD_YOLO_png_640px_rot_rot": 5000, "SDDCB_YOLO_png_640px": 20000, "S2_DETECTION_YOLO_png_640px_rot_rot_spe": 5000, "S2_FC_png_640px_rot_rot_spe": 5000, "SDAI_YOLO_png_640px_rot_rot_spe": 5000, "SRSIN_YOLO_png_640px_rot_rot_spe": 5000 }' --new-dataset D2 --split-ratio "[70,15,15]"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a new dataset from existing ones.')
    parser.add_argument('--dictionary', type=str, required=True, help='Dictionary with the number of images to select per dataset.')
    parser.add_argument('--new-dataset', type=str, required=True, help='Name of the new dataset to create.')
    parser.add_argument('--split-ratio', type=str, required=True, help='List of three numbers representing train, val, test split percentages (e.g., "[70,15,15]").')
    args = parser.parse_args()
    return args

def get_image_counts(dataset_name):
    counts = {}
    total_images = 0
    dataset_path = dataset_name  # Assuming datasets are in current directory
    image_dir = os.path.join(dataset_path, 'images')
    subfolders = ['train', 'val', 'test']
    for subfolder in subfolders:
        folder_path = os.path.join(image_dir, subfolder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist.")
            counts[subfolder] = 0
            continue
        images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        counts[subfolder] = len(images)
        total_images += len(images)
    counts['total'] = total_images
    return counts

def main():
    args = parse_arguments()
    try:
        dataset_counts = ast.literal_eval(args.dictionary)
        if not isinstance(dataset_counts, dict):
            raise ValueError
    except:
        print("Error: Il parametro --dictionary deve essere un dizionario valido, ad esempio: '{\"D1\": 100, \"D2\": 200}'")
        return

    try:
        split_ratio = ast.literal_eval(args.split_ratio)
        if not (isinstance(split_ratio, list) and len(split_ratio) == 3):
            raise ValueError
        split_ratio = [float(x) for x in split_ratio]
        if sum(split_ratio) != 100:
            raise ValueError
    except:
        print("Error: Il parametro --split-ratio deve essere una lista di tre numeri che sommano a 100, ad esempio: \"[70,15,15]\"")
        return

    new_dataset_name = args.new_dataset

    # Initialize dataset_info
    dataset_info = {}
    for dataset_name, num_images in dataset_counts.items():
        if not os.path.exists(dataset_name):
            print(f"Error: Dataset directory {dataset_name} does not exist.")
            return
        counts = get_image_counts(dataset_name)
        counts['required_images'] = num_images
        counts['name'] = dataset_name
        dataset_info[dataset_name] = counts

    # Verify availability
    datasets_below_target = []
    for dataset_name, info in dataset_info.items():
        if info['total'] < info['required_images']:
            shortage = info['required_images'] - info['total']
            datasets_below_target.append({
                'name': info['name'],
                'required_images': info['required_images'],
                'available_images': info['total'],
                'shortage': shortage
            })

    if datasets_below_target:
        print("\nErrore: Non è possibile raggiungere il numero di immagini richiesto per i seguenti dataset:")
        for info in datasets_below_target:
            print(f"Dataset: {info['name']}")
            print(f"  Immagini richieste: {info['required_images']}")
            print(f"  Immagini disponibili: {info['available_images']}")
            print(f"  Immagini mancanti: {info['shortage']}\n")
        print("Lo script terminerà senza generare il nuovo dataset o il file di riepilogo.")
        return

    # Prepare allocations based on split ratio
    subfolders = ['train', 'val', 'test']
    new_dataset_path = new_dataset_name
    for data_type in ['images', 'labels']:
        for subfolder in subfolders:
            dir_path = os.path.join(new_dataset_path, data_type, subfolder)
            os.makedirs(dir_path, exist_ok=True)

    summary_info = {}
    overall_progress = tqdm(total=sum(dataset_counts.values()), desc='Overall Progress')

    for dataset_name, num_images in dataset_counts.items():
        info = dataset_info[dataset_name]
        allocations = {}
        remaining = num_images
        temp_ratios = split_ratio.copy()

        for i, subfolder in enumerate(subfolders):
            if i < len(subfolders) - 1:
                alloc = int(round(num_images * (temp_ratios[i] / 100)))
                alloc = min(alloc, info.get(subfolder, 0))
                allocations[subfolder] = alloc
                remaining -= alloc
            else:
                allocations[subfolder] = remaining  # Assign remaining to the last subfolder

        # Check and adjust if any subfolder doesn't have enough images
        for subfolder in subfolders:
            if allocations[subfolder] > info.get(subfolder, 0):
                shortage = allocations[subfolder] - info.get(subfolder, 0)
                allocations[subfolder] = info.get(subfolder, 0)
                # Redistribute shortage to other subfolders
                for other_subfolder in subfolders:
                    if other_subfolder != subfolder and temp_ratios[subfolders.index(other_subfolder)] > 0:
                        possible_add = info.get(other_subfolder, 0) - allocations[other_subfolder]
                        add = min(shortage, possible_add)
                        allocations[other_subfolder] += add
                        shortage -= add
                        if shortage == 0:
                            break
                if shortage > 0:
                    print(f"Warning: Non è stato possibile allocare tutte le immagini richieste per il dataset {dataset_name} nella sottocartella {subfolder}.")

        dataset_info[dataset_name]['allocations'] = allocations
        summary_info[dataset_name] = allocations

        for subfolder in subfolders:
            num_to_copy = allocations[subfolder]
            if num_to_copy <= 0:
                continue
            source_image_dir = os.path.join(dataset_name, 'images', subfolder)
            source_label_dir = os.path.join(dataset_name, 'labels', subfolder)
            dest_image_dir = os.path.join(new_dataset_path, 'images', subfolder)
            dest_label_dir = os.path.join(new_dataset_path, 'labels', subfolder)
            if not os.path.exists(source_image_dir):
                print(f"Warning: {source_image_dir} non esiste.")
                continue
            images = [f for f in os.listdir(source_image_dir) if os.path.isfile(os.path.join(source_image_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            images = images[:num_to_copy]
            dataset_progress = tqdm(total=len(images), desc=f'Processing {dataset_name} - {subfolder}')
            for image_file in images:
                image_src = os.path.join(source_image_dir, image_file)
                image_dst = os.path.join(dest_image_dir, image_file)
                shutil.copy2(image_src, image_dst)
                # Copy corresponding label
                label_file = os.path.splitext(image_file)[0] + '.txt'
                label_src = os.path.join(source_label_dir, label_file)
                label_dst = os.path.join(dest_label_dir, label_file)
                if os.path.exists(label_src):
                    shutil.copy2(label_src, label_dst)
                else:
                    print(f"Warning: Label file {label_src} non esiste.")
                dataset_progress.update(1)
                overall_progress.update(1)
            dataset_progress.close()

    overall_progress.close()

    # Generate summary
    print("\nRiepilogo:")
    for dataset_name, allocations in summary_info.items():
        total_allocated = sum(allocations.values())
        print(f"Dataset: {dataset_name}")
        print(f"  Totale immagini selezionate: {total_allocated}")
        print(f"  Immagini prese da:")
        for subfolder in subfolders:
            print(f"    {subfolder}: {allocations[subfolder]} immagini")
        contribution = (total_allocated / sum(dataset_counts.values())) * 100
        print(f"  Contributo al totale complessivo: {contribution:.2f}%\n")

    # Generate output .txt file with the same name as the new dataset
    output_file_name = f"{new_dataset_name}.txt"
    output_file = os.path.join(os.path.dirname(os.path.abspath(new_dataset_path)), output_file_name)
    with open(output_file, 'w') as f:
        f.write("Dataset,Train Count,Val Count,Test Count,Total Immagini,Contributo al Totale Complessivo\n")
        for dataset_name, allocations in summary_info.items():
            train_count = allocations['train']
            val_count = allocations['val']
            test_count = allocations['test']
            total_images = sum(allocations.values())
            contribution = (total_images / sum(dataset_counts.values())) * 100
            f.write(f"{dataset_name},{train_count},{val_count},{test_count},{total_images},{contribution:.2f}%\n")
        f.write(f"Totale complessivo di immagini in tutti i dataset,{sum(dataset_counts.values())},,,,\n")
        total_train_images = sum([allocations['train'] for allocations in summary_info.values()])
        total_val_images = sum([allocations['val'] for allocations in summary_info.values()])
        total_test_images = sum([allocations['test'] for allocations in summary_info.values()])
        f.write(f"Totale complessivo immagini per Train,{total_train_images},,,,\n")
        f.write(f"Totale complessivo immagini per Val,{total_val_images},,,,\n")
        f.write(f"Totale complessivo immagini per Test,{total_test_images},,,,\n")

    print(f"\nRiepilogo scritto nel file {output_file}")

    # Additional check: Verify that every image has a corresponding label
    print("\nVerifica delle corrispondenze tra immagini e label nel dataset generato:")
    missing_labels = []
    for subfolder in subfolders:
        image_dir = os.path.join(new_dataset_path, 'images', subfolder)
        label_dir = os.path.join(new_dataset_path, 'labels', subfolder)
        images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in images:
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(label_path):
                missing_labels.append({
                    'image': os.path.join(image_dir, image_file),
                    'expected_label': label_path
                })

    if missing_labels:
        print(f"\nTrovate {len(missing_labels)} immagini senza label corrispondente:")
        for item in missing_labels:
            print(f"Immagine senza label: {item['image']}")
            print(f"Label attesa: {item['expected_label']}\n")
    else:
        print("Tutte le immagini hanno una label corrispondente.")

if __name__ == '__main__':
    main()
