import os

# Создание папки для сохранения результатов обучения
def dir():

    base_dir = os.getcwd()
    base_dir_runs = os.path.join(base_dir, "runs")
    if not os.path.exists(base_dir_runs):
        os.makedirs(base_dir_runs)

    train_folders = []
    for folder in os.listdir(base_dir_runs):
        if os.path.isdir(os.path.join(base_dir_runs, folder)) and folder.startswith("train_"):
            train_folders.append(folder)

    folder_numbers = []
    for f in train_folders:
        try:
            num = int(f.split("_")[-1])
            folder_numbers.append(num)
        except ValueError:
            continue

    next_number = max(folder_numbers) + 1 if folder_numbers else 0
    new_folder = f"train_{next_number}"
    new_folder_path = os.path.join(base_dir_runs, new_folder)

    os.makedirs(new_folder_path, exist_ok=True)
    print(f"Создана папка: {new_folder_path}")

    return new_folder_path
