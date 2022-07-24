set -ex

python3 train.py --batch_size 1 --sample_batch_size 1 --log_rate 1 --checkpoint_rate 1 --debug True --log_to_wandb False --num_timesteps 2 --log_dir ddpm_logs/ --use_mnist False --use_cifar True --img_channels 3 --initial_pad 0 --img_size 32 --project_name 'diffusion-models-cifar'
