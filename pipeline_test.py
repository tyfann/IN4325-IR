import os
import sys

import json


# Get the name of the config folder from the user

def config_generate():
    loss_group = ["approxndcg", "neuralndcg", "ranknet", "mse"]
    # Loop through the config files
    for loss in loss_group:
        config_folder = "allrank\configs"
        config_files = [f for f in os.listdir(config_folder) if
                        f.startswith('config_{}'.format(loss)) and f.endswith('.json')]
        for config_file in config_files:
            config_file_path = os.path.join(config_folder, config_file)
            # 读取原始JSON文件
            with open(config_file_path, "r") as f:
                config = json.load(f)

            fold_common_name = "Fold"
            fold_names = ["{}{}_normalized".format(fold_common_name, i) for i in range(1, 6, 1)]
            print(fold_names)
            for fold_name in fold_names:
                value = "C:\\Users\\tyfann\\Documents\\TU Delft\\IN4325 Information Retrieval\\IN4325-IR\\data\\MSLR-WEB10K\\{}".format(
                    fold_name)
                config["data"]["path"] = value
                new_config_file = config_file.replace(config_file.split("_")[3].split(".")[0], fold_name.split("_")[0].lower())

                new_filename = os.path.join(config_folder, new_config_file)
                with open(new_filename, "w") as f:
                    json.dump(config, f, indent=4)


if __name__ == '__main__':
    config_generate()
    exit(0)
    config_folder = "allrank\configs"
    loss_group = ["approxndcg", "neuralndcg", "ranknet", "mse"]
    # Loop through the config files
    for loss in loss_group:
        # Build the path to the config file
        config_files = [f for f in os.listdir(config_folder) if
                        f.startswith('config_{}'.format(loss)) and f.endswith('.json')]
        # config_file = os.path.join(config_folder, "config_{}.json".format(i))
        # print(config_files)
        for config_file_name in config_files:
            task_name = config_file_name.split("_")[2]
            fold_name = config_file_name.split("_")[3].split(".")[0]
            run_id = "test_run_{}".format(loss)
            job_dir = os.path.join(r"allrank\task-{}".format(task_name), "{}-lr0001-{}".format(loss, fold_name))
            print(
                "python allrank\main.py --config-file-name {} --run-id {} --job-dir {}".format(config_file_name, run_id,
                                                                                               job_dir))
            # os.system("python allrank\main.py --config-file-name {} --run-id {} --job-dir {}".format(config_file_name, run_id, job_dir))
        break
    # # Extract the file number
    # file_number = str(i)
    # file_number = file_number.replace("config", "")
    #
    # # Call the main script with the config file and file number as arguments
    # os.system("python main.py {} {}".format(config_file, file_number))
