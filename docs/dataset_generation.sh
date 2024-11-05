# define dataset folder
export DATASET_NAME=leap_test1
# define mesh path
export MESHDATA_ROOT_PATH=data/meshdata

# define number of objects to use
python get_a_grip/dataset_generation/scripts/generate_object_code_and_scales_txt.py \
--meshdata_root_path ${MESHDATA_ROOT_PATH} \
--output_object_code_and_scales_txt_path data/dataset/${DATASET_NAME}/object_code_and_scales.txt \
--min_object_scale 0.05 \
--max_object_scale 0.1 \
--num_scales_per_object 3 \
--max_num_object_codes 3 \

# drop object on table
# log successful objects
python get_a_grip/dataset_generation/scripts/generate_nerfdata.py \
--meshdata_root_path ${MESHDATA_ROOT_PATH} \
--input_object_code_and_scales_txt_path data/dataset/${DATASET_NAME}/object_code_and_scales.txt \
--output_nerfdata_path data/dataset/${DATASET_NAME}/nerfdata \
--num_cameras 10

# compute hand configs
python get_a_grip/dataset_generation/scripts/generate_hand_config_dicts.py \
--meshdata_root_path ${MESHDATA_ROOT_PATH} \
--input_object_code_and_scales_txt_path data/dataset/${DATASET_NAME}/nerfdata_settled_successes.txt \
--output_hand_config_dicts_path data/dataset/${DATASET_NAME}/hand_config_dicts \
--hand_model_type LEAP

# compute grasp configs
python get_a_grip/dataset_generation/scripts/generate_grasp_config_dicts.py \
--meshdata_root_path ${MESHDATA_ROOT_PATH} \
--input_hand_config_dicts_path data/dataset/${DATASET_NAME}/hand_config_dicts \
--output_grasp_config_dicts_path data/dataset/${DATASET_NAME}/grasp_config_dicts \
--hand_model_type LEAP

# evaluate grasps
python get_a_grip/dataset_generation/scripts/eval_all_grasp_config_dicts.py \
--meshdata_root_path ${MESHDATA_ROOT_PATH} \
--input_grasp_config_dicts_path data/dataset/${DATASET_NAME}/grasp_config_dicts \
--output_evaled_grasp_config_dicts_path data/dataset/${DATASET_NAME}/evaled_grasp_config_dicts \
# num_random_noise_sample_per_grasp is None by default

