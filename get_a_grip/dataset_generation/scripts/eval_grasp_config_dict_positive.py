# gymtorch must be imported before torch (thus isaac_validator should be first)
# isort: off
from get_a_grip.dataset_generation.utils.isaac_validator import (
    IsaacValidator,
    ValidationType,
)
import torch
# isort: on

import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import tyro
from tqdm import tqdm

from get_a_grip import get_data_folder
from get_a_grip.dataset_generation.utils.hand_model import HandModel, HandModelType
from get_a_grip.dataset_generation.utils.joint_angle_targets import (
    compute_init_joint_angles_given_grasp_orientations,
    compute_optimized_joint_angle_targets_given_grasp_orientations,
)
from get_a_grip.dataset_generation.utils.pose_conversion import (
    hand_config_np_to_pose,
)
from get_a_grip.dataset_generation.utils.torch_quat_utils import matrix_to_quat_wxyz
from get_a_grip.utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
from get_a_grip.utils.seed import set_seed

@dataclass
class EvalGraspConfigDictArgs:
    meshdata_root_path: pathlib.Path = get_data_folder() / "meshdata"
    input_grasp_config_dicts_path: pathlib.Path = (
        get_data_folder() / "dataset/NEW/grasp_config_dicts"
    )
    output_evaled_grasp_config_dicts_path: Optional[pathlib.Path] = (
        get_data_folder() / "dataset/NEW/evaled_grasp_config_dicts_positive"
    )
    object_code_and_scale_str: str = (
        "core-bottle-2722bec1947151b86e22e2d2f64c8cef_0_1000"
    )
    validation_type: ValidationType = ValidationType.GRAVITY_AND_TABLE
    hand_model_type: HandModelType = HandModelType.ALLEGRO
    gpu: int = 0
    max_grasps_per_batch: int = 5000  # Reasonable default, but some extreme objects require more GPU memory, so this can be lowered for those
    move_fingers_back_at_init: bool = False

    # if debug_index is received, then the debug mode is on
    debug_index: Optional[int] = None
    start_with_step_mode: bool = False  # with use_gui, starts sim paused in step mode, press S to step 1 sim step, press space to toggle pause
    use_gui: bool = False
    use_cpu: bool = False  # GPU is faster. NOTE: there can be discrepancies between using GPU vs CPU, likely different slightly physics
    save_to_file: bool = True
    record_indices: List[int] = field(default_factory=list)

def compute_joint_angle_targets(
    hand_pose: torch.Tensor,
    grasp_orientations: torch.Tensor,
    hand_model_type: HandModelType,
) -> np.ndarray:
    grasp_orientations = grasp_orientations.to(hand_pose.device)

    # hand model
    hand_model = HandModel(hand_model_type=hand_model_type, device=hand_pose.device)
    hand_model.set_parameters(hand_pose)
    assert hand_model.hand_pose is not None

    # Optimization
    (
        optimized_joint_angle_targets,
        _,
    ) = compute_optimized_joint_angle_targets_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=grasp_orientations,
    )

    num_joints = hand_model.num_joints
    assert optimized_joint_angle_targets.shape == (hand_model.batch_size, num_joints)

    return optimized_joint_angle_targets.detach().cpu().numpy()

def compute_init_joint_angles(
    hand_pose: torch.Tensor,
    grasp_orientations: torch.Tensor,
    hand_model_type: HandModelType,
) -> np.ndarray:
    grasp_orientations = grasp_orientations.to(hand_pose.device)

    # hand model
    hand_model = HandModel(hand_model_type=hand_model_type, device=hand_pose.device)
    hand_model.set_parameters(hand_pose)
    assert hand_model.hand_pose is not None

    # Optimization
    (
        init_joint_angles,
        _,
    ) = compute_init_joint_angles_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=grasp_orientations,
    )

    num_joints = hand_model.num_joints
    assert init_joint_angles.shape == (hand_model.batch_size, num_joints)

    return init_joint_angles.detach().cpu().numpy()

def eval_grasp_config_dict(args: EvalGraspConfigDictArgs) -> dict:
    if args.save_to_file:
        assert (
            args.output_evaled_grasp_config_dicts_path is not None
        ), "output_evaled_grasp_config_dicts_path must be set if save_to_file is True"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # no need for N_NOISY since we want only positive samples

    object_code, object_scale = parse_object_code_and_scale(
        args.object_code_and_scale_str
    )
    set_seed(42)  # Want this fixed so deterministic computation

    # Read in data
    grasp_config_dict_path = (
        args.input_grasp_config_dicts_path / f"{args.object_code_and_scale_str}.npy"
    )
    print(f"Loading grasp config dicts from: {grasp_config_dict_path}")
    grasp_config_dict: Dict[str, Any] = np.load(
        grasp_config_dict_path, allow_pickle=True
    ).item()
    trans: np.ndarray = grasp_config_dict["trans"]
    rot: np.ndarray = grasp_config_dict["rot"]
    joint_angles: np.ndarray = grasp_config_dict["joint_angles"]
    grasp_orientations: np.ndarray = grasp_config_dict["grasp_orientations"]

    # Compute hand pose
    quat_wxyz = matrix_to_quat_wxyz(torch.from_numpy(rot)).numpy()
    hand_pose = hand_config_np_to_pose(
        trans=trans, rot=rot, joint_angles=joint_angles
    ).to(device)

     # Compute joint angle targets
    joint_angle_targets_array = compute_joint_angle_targets(
        hand_pose=hand_pose,
        grasp_orientations=torch.from_numpy(grasp_orientations).float().to(device),
        hand_model_type=args.hand_model_type,
    )
    init_joint_angles = (
        compute_init_joint_angles(
            hand_pose=hand_pose,
            grasp_orientations=torch.from_numpy(grasp_orientations).float().to(device),
            hand_model_type=args.hand_model_type,
        )
        if args.move_fingers_back_at_init
        else joint_angles
    )

    #TODO: add debugger (rayliu)

    sim = IsaacValidator(
        gpu=args.gpu,
        validation_type=args.validation_type,
        hand_model_type=args.hand_model_type,
        mode="gui" if args.use_gui else "headless",
        start_with_step_mode=args.start_with_step_mode,
        use_cpu=args.use_cpu,
    )
    # Run validation on all grasps
    batch_size = trans.shape[0]
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)

    # Some final shape checking.
    assert quat_wxyz.shape == (batch_size, 4)
    assert joint_angles.shape == (batch_size, 16)
    assert hand_pose.shape == (batch_size, 3 + 6 + 16)
    assert grasp_orientations.shape == (batch_size, hand_model.num_fingers, 3, 3)
    assert joint_angle_targets_array.shape == (batch_size, 16)
    assert init_joint_angles.shape == (batch_size, 16)

    # Run for loop over minibatches of grasps.
    y_pick_array = []
    y_coll_object_array = []
    y_coll_table_array = []
    object_states_before_grasp_array = []
    max_grasps_per_batch = args.max_grasps_per_batch
    pbar = tqdm(
        range(math.ceil(batch_size / max_grasps_per_batch)),
        desc="evaling batches of grasps",
    )

    for i in pbar:
        start_index = i * max_grasps_per_batch
        end_index = min((i + 1) * max_grasps_per_batch, batch_size)
        sim.set_obj_asset(
            obj_root=str(args.meshdata_root_path / object_code / "coacd"),
            obj_file="coacd.urdf",
        )
    
    for index in range(start_index, end_index):
        sim.add_env(
            hand_quaternion_wxyz=quat_wxyz[index],
            hand_translation=trans[index],
            hand_qpos=init_joint_angles[index],
            obj_scale=object_scale,
            target_qpos=joint_angle_targets_array[index],
            # add_random_pose_noise is False by default
            record=index in args.record_indices,
        )

        (
            y_pick,
            y_coll_object,
            y_coll_table,
            object_states_before_grasp,
        ) = sim.run_sim()
        y_pick_array.extend(y_pick)
        y_coll_object_array.extend(y_coll_object)
        y_coll_table_array.extend(y_coll_table)
        object_states_before_grasp_array.append(
            object_states_before_grasp.reshape(-1, 13)
        )

        sim.reset_simulator()
        pbar.set_description(
            f"evaling batches of grasps: mean_success = {np.mean(y_pick_array)}"
        )

        hand_model.set_parameters(hand_pose[start_index:end_index])

    # Aggregate results
    y_pick_array = np.array(y_pick_array)
    y_coll_object_array = np.array(y_coll_object_array)
    y_coll_table_array = np.array(y_coll_table_array)
    object_states_before_grasp_array = np.concatenate(
        object_states_before_grasp_array, axis=0
    )

    assert y_pick_array.shape == (batch_size,)
    assert y_coll_object_array.shape == (batch_size,)
    assert y_coll_table_array.shape == (batch_size,)

    object_states_before_grasp_array = object_states_before_grasp_array.reshape(batch_size, 1, 13)

    y_coll_array = y_coll_object_array * y_coll_table_array

    y_PGS = y_pick_array * y_coll_array

    # TODO: add debug section (rayliu)

    sim_frac = np.mean(y_pick_array)
    new_pen_frac = np.mean(y_coll_array)
    eval_frac = np.mean(y_PGS)
    print("=" * 80)
    print(
        f"y_pick: {y_pick_array.sum().item()}/{batch_size} ({100 * sim_frac:.2f}%), "
        f"y_coll: {y_coll_array.sum().item()}/{batch_size} ({100 * new_pen_frac:.2f}%), "
        f"y_PGS = y_pick*y_coll: {y_PGS.sum().item()}/{batch_size} ({100 * eval_frac:.2f}%)"
    )
    print("=" * 80)

    # logged in the evalued_grasp_config_dicts_positive
    evaled_grasp_config_dict = {
        **grasp_config_dict,
        "y_coll": y_coll_array,
        "y_pick": y_pick_array,
        "y_PGS": y_PGS,
        "object_states_before_grasp": object_states_before_grasp_array,
    }

    if args.save_to_file:
        assert args.output_evaled_grasp_config_dicts_path is not None
        print(
            f"Saving evaled grasp config dicts to: {args.output_evaled_grasp_config_dicts_path}"
        )
        args.output_evaled_grasp_config_dicts_path.mkdir(parents=True, exist_ok=True)
        np.save(
            file=(
                args.output_evaled_grasp_config_dicts_path
                / f"{args.object_code_and_scale_str}.npy"
            ),
            arr=evaled_grasp_config_dict,
            allow_pickle=True,
        )
        print(f"Done saving to: {args.output_evaled_grasp_config_dicts_path}")

    print("Attempting to destroy sim...")
    sim.destroy()
    print("Destroyed sim!")
    return evaled_grasp_config_dict

def main() -> None:
    args = tyro.cli(tyro.conf.FlagConversionOff[EvalGraspConfigDictArgs])
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    eval_grasp_config_dict(args)


if __name__ == "__main__":
    main()