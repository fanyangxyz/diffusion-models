set -ex

python3 train_mnist.py --batch_size 1 --sample_batch_size 1 --log_rate 1 --checkpoint_rate 1 --debug True --log_to_wandb False --num_timesteps 2 --log_dir ddpm_logs/
