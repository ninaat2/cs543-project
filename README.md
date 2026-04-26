# cs543-project
cd ~/cs543-project
pip install gsplat pycolmap==3.12.6 tyro viser splines nerfview tensorboard imageio imageio-ffmpeg scipy plyfile
pip install opencv-python
python simple_trainer.py default \
    --data_dir glass/ \
    --data_factor 1 \
    --result_dir ./results/glass

