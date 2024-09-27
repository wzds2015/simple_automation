## Check connection with iPhone

1. Install mujoco:

```bash
pip install mujoco
```

2. Install mink-mujocoAR

```bash
git clone https://github.com/omarrayyann/mink-mujocoAR
```

3. Install app from AppStore https://apps.apple.com/ae/app/mujoco-ar/id6612039501

4. Connect iPhone and laptop to one wifi network. Run script, file shown address in app and have fun controlling robot in simulation with iPhone.

```bash
cd mink-mujocoAR
mjpython arm_ur5e.py
```

#### Troubleshooting

-   You can share hotspot in iPhone and connect laptop to it. iPhone should be connected to some wifi network (not cellular) at that time.

## Recording episodes for training

1. Install Lerobot in another folder following the installation section:
   https://github.com/huggingface/lerobot

2. Run teleoperation and control robot with MuJoCo AR app from previous section (low cost robot)

```bash
mjpython control_simulated_robot.py teleoperate --robot_name lerobot
```

3. Run episode recording (ur5e)

```bash
mjpython control_simulated_robot.py record --robot_name ur5e
```

## Visualise episode

Go to Lerobot folder and run command bellow. Don't forget to change absolute address

```bash
python lerobot/scripts/visualize_dataset_html.py \
  --root /Users/igor/Documents/_Develop/_Robotics/simple_automation/data \
  --repo-id 1g0rrr/test_painting
```

## Training

Go to Lerobot folder
Run this command. Don't forget to change DATA_DIR and --config-dir for your locations.

```bash
DATA_DIR=/Users/igor/Documents/_Develop/_Robotics/simple_automation/data python lerobot/scripts/train.py \
 --config-dir=/Users/igor/Documents/_Develop/_Robotics/simple_automation/configs \
  dataset_repo_id=1g0rrr/test_painting \
  policy=act_simple_automation \
  env=simple_automation \
  hydra.run.dir=outputs/train/test_painting \
  hydra.job.name=act_test_painting \
  device=mps \
  wandb.enable=false
```
