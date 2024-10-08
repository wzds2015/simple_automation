import argparse
import concurrent.futures
import json
import logging
import os
import platform
import shutil
import time
import traceback
from contextlib import nullcontext
from functools import cache
from pathlib import Path
import mujoco
import mujoco.viewer
import mink
from mujoco_ar import MujocoARConnector
from loop_rate_limiters import RateLimiter
import math
import numpy as np

import cv2
import torch
import tqdm
from omegaconf import DictConfig
from PIL import Image
from termcolor import colored

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index, create_branch
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
# from lerobot.common.robot_devices.robots.sim_robot import SimRobot, SimRobotConfig
from lerobot.common.robot_devices.robots.utils import Robot, get_arm_id
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.scripts.push_dataset_to_hub import (
    push_dataset_card_to_hub,
    push_meta_data_to_hub,
    push_videos_to_hub,
    save_meta_data,
)
from iphone_leader import IPhoneLeader
from simulated_follower import SimulatedFollower
from sim_camera import SimCamera

def save_image(img_tensor, key, frame_index, episode_index, videos_dir):
    img = Image.fromarray(img_tensor.numpy())
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)

def none_or_int(value):
    if value == "None":
        return None
    return int(value)

def resetColor(model, data):
    initial_color = [0.0, 0.5, 0, 1]  # RGBA format 

    for i in range(1, 10):
        model.geom_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"green_box{i}")] = initial_color
    
def checkCollisions(model, data):
    mujoco.mj_collision(model, data)

    new_color = [1.0, 0.0, 0.0, 1.0]  # RGBA format 
    # Check collision with specific objects

    for i in range(data.ncon):
        try:
            contact = data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            body1_id = model.geom_bodyid[geom1_id]
            body2_id = model.geom_bodyid[geom2_id]

            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            if (body1_name == "wrist_3_link" and body2_name == "surface") or (body1_name == "surface" and body2_name == "wrist_3_link"):
                # print(f"Collision detected between {geom1_name} and {geom2_name}")
                if body1_name == "surface":
                    # print(geom1_id)
                    model.geom_rgba[geom1_id] = new_color
                elif body2_name == "surface":
                    # print(geom1_id)
                    model.geom_rgba[geom2_id] = new_color
        except Exception as e:
            print(e)    


def teleoperate(robot: Robot, teleop_time_s: float | None = None):

    # frame_name="static_side",
    # frame_type="geom",

    robot.connect()


    # TELEOPERATE

    sim_fps = 30
    teleop_fps = 30
    sim_teleop_ratio = math.ceil(sim_fps / teleop_fps)

    dt_s = 1 / sim_fps

    counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while True:
            start_loop_t = time.perf_counter()

            if counter % sim_teleop_ratio == 0:
                robot.teleop_step()

            checkCollisions(model, data)

            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            viewer.sync()
            # mujoco.mj_step(model, data)

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / sim_fps - dt_s)


def record(robot: Robot):
    policy: torch.nn.Module | None = None
    repo_id="1g0rrr/test_painting"
    root="data"
    force_override=True
    warmup_time_s=5
    episode_time_s=30
    reset_time_s=5
    num_episodes=1
    num_image_writers_per_camera=4
    device="cpu"
    seed = 1000
    video=True
    run_compute_stats=True
    push_to_hub=False

    _, dataset_name = repo_id.split("/")

    robot.connect()

    local_dir = Path(root) / repo_id
    if local_dir.exists() and force_override:
        shutil.rmtree(local_dir)

    episodes_dir = local_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = local_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Logic to resume data recording
    rec_info_path = episodes_dir / "data_recording_info.json"
    if rec_info_path.exists():
        with open(rec_info_path) as f:
            rec_info = json.load(f)
        episode_index = rec_info["last_episode_index"] + 1
    else:
        episode_index = 0


    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    exit_early = False
    rerecord_episode = False
    stop_recording = False

    from pynput import keyboard

    def on_press(key):
        nonlocal exit_early, rerecord_episode, stop_recording
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                exit_early = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                rerecord_episode = True
                exit_early = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                stop_recording = True
                exit_early = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()


    sim_fps = 30
    teleop_fps = 30
    sim_teleop_ratio = math.ceil(sim_fps / teleop_fps)

    dt_s = 1 / sim_fps

    counter = 0


    # Execute a few seconds without recording data, to give times
    # to the robot devices to connect and start synchronizing.


    # Save images using threads to reach high fps (30 and more)
    # Using `with` to exist smoothly if an execption is raised.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)


        timestamp = 0
        start_warmup_t = time.perf_counter()
        is_warmup_print = False
        while timestamp < warmup_time_s:
            if not is_warmup_print:
                print("Warming up")
                is_warmup_print = True

            start_loop_t = time.perf_counter()

            if counter % sim_teleop_ratio == 0:
                if policy is None:
                    observation, action = robot.teleop_step(record_data=True)
                else:
                    observation = robot.capture_observation()

            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            viewer.sync()

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / sim_fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t

            timestamp = time.perf_counter() - start_warmup_t


        futures = []
        num_image_writers = num_image_writers_per_camera * len(robot.cameras)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_image_writers) as executor:
            # Start recording all episodes
            while episode_index < num_episodes:
                print(f"Recording episode {episode_index}")
                ep_dict = {}
                frame_index = 0
                timestamp = 0
                start_episode_t = time.perf_counter()

                resetColor(model, data)
                viewer.sync()

                while timestamp < episode_time_s:
                    start_loop_t = time.perf_counter()

                    viewer.sync()

                    if counter % sim_teleop_ratio == 0:
                        observation, action = robot.teleop_step(record_data=True)


                        image_keys = [key for key in observation if "image" in key]
                        not_image_keys = [key for key in observation if "image" not in key]

                        for key in image_keys:
                            futures += [
                                executor.submit(
                                    save_image, observation[key], key, frame_index, episode_index, videos_dir
                                )
                            ]

                        image_keys = [key for key in observation if "image" in key]

                        for key in not_image_keys:
                            if key not in ep_dict:
                                ep_dict[key] = []
                            ep_dict[key].append(observation[key])

                        for key in action:
                            if key not in ep_dict:
                                ep_dict[key] = []
                            ep_dict[key].append(action[key])

                        frame_index += 1

                    checkCollisions(model, data)

                    mujoco.mj_camlight(model, data)

                    # mujoco.mj_fwdPosition(model, data)
                    mujoco.mj_sensorPos(model, data)
                    mujoco.mj_forward(model, data)


                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / sim_fps - dt_s)

                    timestamp = time.perf_counter() - start_episode_t
                    if exit_early:
                        exit_early = False
                        break

                print(f"Episode {episode_index} done")

                if not stop_recording:
                    # Start resetting env while the executor are finishing
                    logging.info("Reset the environment")

                timestamp = 0
                start_vencod_t = time.perf_counter()

                # During env reset we save the data and encode the videos
                num_frames = frame_index

                for key in image_keys:
                    # tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
                    fname = f"{key}_episode_{episode_index:06d}.mp4"
                    video_path = local_dir / "videos" / fname
                    if video_path.exists():
                        video_path.unlink()
                    # Store the reference to the video frame, even tho the videos are not yet encoded
                    ep_dict[key] = []
                    for i in range(num_frames):
                        ep_dict[key].append({"path": f"videos/{fname}", "timestamp": i / teleop_fps})

                for key in not_image_keys:
                    ep_dict[key] = torch.stack(ep_dict[key])

                for key in action:
                    ep_dict[key] = torch.stack(ep_dict[key])

                ep_dict["episode_index"] = torch.tensor([episode_index] * num_frames)
                ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
                ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / teleop_fps

                done = torch.zeros(num_frames, dtype=torch.bool)
                done[-1] = True
                ep_dict["next.done"] = done

                ep_path = episodes_dir / f"episode_{episode_index}.pth"
                print("Saving episode dictionary...")
                torch.save(ep_dict, ep_path)

                rec_info = {
                    "last_episode_index": episode_index,
                }
                with open(rec_info_path, "w") as f:
                    json.dump(rec_info, f)

                is_last_episode = stop_recording or (episode_index == (num_episodes - 1))

                # Wait if necessary
                with tqdm.tqdm(total=reset_time_s, desc="Waiting") as pbar:
                    while timestamp < reset_time_s and not is_last_episode:
                        time.sleep(1)
                        timestamp = time.perf_counter() - start_vencod_t
                        pbar.update(1)
                        if exit_early:
                            exit_early = False
                            break

                # Skip updating episode index which forces re-recording episode
                if rerecord_episode:
                    rerecord_episode = False
                    continue

                episode_index += 1

                if is_last_episode:
                    print("Done recording")
                    listener.stop()

                    logging.info("Waiting for threads writing the images on disk to terminate...")
                    for _ in tqdm.tqdm(
                        concurrent.futures.as_completed(futures), total=len(futures), desc="Writting images"
                    ):
                        pass
                    break

    robot.disconnect()
    cv2.destroyAllWindows()
# 
# 
    num_episodes = episode_index

    print("Encoding videos")
    # Use ffmpeg to convert frames stored as png into mp4 videos
    for episode_index in tqdm.tqdm(range(num_episodes)):
        for key in image_keys:
            tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            video_path = local_dir / "videos" / fname
            if video_path.exists():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
            # since video encoding with ffmpeg is already using multithreading.
            encode_video_frames(tmp_imgs_dir, video_path, teleop_fps, vcodec="libx264", overwrite=True)
            shutil.rmtree(tmp_imgs_dir)

    ep_dicts = []
    for episode_index in tqdm.tqdm(range(num_episodes)):
        ep_path = episodes_dir / f"episode_{episode_index}.pth"
        ep_dict = torch.load(ep_path)
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": teleop_fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    if run_compute_stats:
        print("Computing dataset statistics")
        stats = compute_stats(lerobot_dataset, num_workers=4)
        lerobot_dataset.stats = stats
    else:
        stats = {}
        logging.info("Skipping computation of the dataset statistics")

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    meta_data_dir = local_dir / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    # if push_to_hub:
    #     hf_dataset.push_to_hub(repo_id, revision="main")
    #     push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
    #     push_dataset_card_to_hub(repo_id, revision="main", tags=None)
    #     if video:
    #         push_videos_to_hub(repo_id, videos_dir, revision="main")
    #     create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

    print("Done")
    return lerobot_dataset


def replay(robot: Robot):
    pass

def evaluate(robot:Robot):
    global follower
    global leader
    global _viewer
    global repo_id
    global fps

    inference_time_s = 610
    fps = 30
    device = "cpu" # "cuda" or "cpu"
    robot.connect()

    from lerobot.common.policies.act.modeling_act import ACTPolicy
    ckpt_path = "outputs/train/test_painting/checkpoints/last/pretrained_model"
    policy = ACTPolicy.from_pretrained(ckpt_path)

    policy.to(device)

    evaluate_time = 60


    sim_fps = 30
    teleop_fps = 30
    sim_teleop_ratio = math.ceil(sim_fps / teleop_fps)

    dt_s = 1 / sim_fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        timestamp = 0
        start_evaluate_t = time.perf_counter()
        while timestamp < evaluate_time:

            start_loop_t = time.perf_counter()

            # if counter % sim_teleop_ratio == 0:
            observation = robot.capture_observation()

            # Convert to pytorch format: channel first and float32 in [0,1]
            # with batch dimension
            for name in observation:
                if "image" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()
                observation[name] = observation[name].unsqueeze(0)
                observation[name] = observation[name].to(device)

            # Compute the next action with the policy
            # based on the current observation
            observation["observation.state"] = observation["observation.state"].float()
            action = policy.select_action(observation)
            # # Remove batch dimension
            action = action.squeeze(0)
            # # Move to cpu, if not already the case
            action = action.to("cpu")
            # # Order the robot to move
            robot.send_action(action)
            mujoco.mj_forward(model, data)
            viewer.sync()

            mujoco.mj_collision(model, data)
            checkCollisions(model, data)

            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)


            timestamp = time.perf_counter() - start_evaluate_t

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / sim_fps - dt_s)

def fix_stats():

    repo_id="1g0rrr/test_painting"
    root="data"

    local_dir = Path(root) / repo_id

    lerobot_dataset = LeRobotDataset(repo_id, root, "train")

    stats = compute_stats(lerobot_dataset, num_workers=4)
    lerobot_dataset.stats = stats

    meta_data_dir = local_dir / "meta_data"
    teleop_fps = 30

    hf_dataset = lerobot_dataset.hf_dataset
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": teleop_fps,
        "video": True,
    }
    info["encoding"] = get_default_encoding()    
    save_meta_data(info, stats, episode_data_index, meta_data_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=False)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot_name",
        type=str,
        default="ur5e",
        help="Name of the robot",
    )

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument(
        "--teleop-time-s",
        type=int,
        default=10,
        help="Number of seconds.",
    )
        
    parser_record = subparsers.add_parser("record", parents=[base_parser])

    parser_replay = subparsers.add_parser("replay", parents=[base_parser])
    parser_replay = subparsers.add_parser("stats", parents=[base_parser])
    parser_replay = subparsers.add_parser("evaluate", parents=[base_parser])
    args = parser.parse_args()

    if args.robot_name == "ur5e":
        path_scene="assets/universal_robots_ur5e/scene.xml"
        robot_name = "ur5e"
        frame_name = "attachment_site"
        frame_type = "site"
    elif args.robot_name == "lerobot":
        path_scene="assets/low_cost_robot_6dof/pick_place_cube.xml"
        robot_name = "lerobot"
        # frame_name = "gripper_assembly"
        # frame_type = "body"
        frame_name = "end_effector"
        frame_type = "geom"
    else:
        raise KeyError(f"Robot name not support: {args.robot_name}")

    model = mujoco.MjModel.from_xml_path(path_scene)
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)
    model = configuration.model
    data = configuration.data
    
    leader_arm = IPhoneLeader(
        configuration,
        frame_name=frame_name,
        frame_type=frame_type,
        robot_name=robot_name,
    )
    follower_arm = SimulatedFollower(configuration)
    

    cameras = {
        "image_top":   SimCamera(id_camera="camera_top",   model=model, data=data, camera_index=0, fps=30, width=640, height=480),
        "image_front": SimCamera(id_camera="camera_front", model=model, data=data, camera_index=1, fps=30, width=640, height=480),
    }


    robot = ManipulatorRobot(
        leader_arms={"main": leader_arm},
        follower_arms={"main": follower_arm},
        calibration_dir="cache/calibration/mysim",
        robot_type="mysim",
        cameras=cameras,
    )

    control_mode = args.mode
    kwargs = vars(args)
    del kwargs["mode"]

    print(f"Control mode: {control_mode}")

    try:
        if control_mode == "teleoperate":
            teleoperate(robot=robot, teleop_time_s=args.teleop_time_s)

        elif control_mode == "record":
            record(robot=robot)

        elif control_mode == "replay":
            replay(robot=robot) 
        elif control_mode == "stats":
            fix_stats()
        elif control_mode == "evaluate":
            evaluate(robot=robot)
    except Exception as e:
        print(f"Error: {e}")
        robot.disconnect()
        pass


    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()




