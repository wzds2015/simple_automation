## Check connection with iPhone without ACT

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

2. Run teleoperation and control robot with MuJoCo AR app from previous section.

```bash
mjpython control_simulated_robot.py teleoperate
```

3. Run episode recording

```bash
mjpython control_simulated_robot.py record
```
