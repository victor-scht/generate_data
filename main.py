import generate_data.utils_generate as utils_generate
import generate_data.generate_data as generate
import data_analysis.generate_info as generate_info
import yaml
from datetime import datetime
import os
import argparse

# =========================== parse arguments

# to save a new config
parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

save_status = args.save

# =========================== load and/or save a new config

rel_path = "config/config.yaml"
current_dir = os.getcwd()

conf_path = os.path.join(current_dir,rel_path)

with open(conf_path, "r") as file :
    config = yaml.safe_load(file)

location,name0 = config["location"], config["directory_name"]

now = datetime.now()
formatted_date = now.strftime("-%Y-%m-%d:%H:%M:%S")
name = name0 + formatted_date

if save_status: 
    conf_path = os.path.join(current_dir,"config/saved",name+".yaml")
    folder_path = os.path.dirname(conf_path)
    folder = os.listdir(folder_path)
    if len(folder) > 10 : 
        file_name = folder[0]
        file_path = os.path.join(folder_path,file_name)
        os.remove(file_path)

    with open(conf_path,"w") as file :
        yaml.dump(config,file)


# =========================== generate data

directory_structure = utils_generate.get_directory_paths(location,name0,name)
utils_generate.create_directory(directory_structure)

L=["train", "test", "val"]

for purpose in L : 
    generate.gen_patches(config,directory_structure,purpose)

# =========================== generate_info

generate_info.generate_info_snr(directory_structure,"train")
generate_info.generate_info_ps_pn(directory_structure,"train")
generate_info.plot(directory_structure,"train")
generate_info.copy(directory_structure,name)
