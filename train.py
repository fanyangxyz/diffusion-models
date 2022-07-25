import argparse
import datetime
import torch
import wandb

from torch.utils.data import DataLoader
from torchvision import datasets
from ddpm import script_utils


def main():
    args = create_argparser().parse_args()
    assert not (
        args.use_mnist and args.use_cifar), "Cannot use MNIST and CIFAR both."
    assert args.use_mnist or args.use_cifar, "Must use either MNIST or CIFAR."
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(
            diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError(
                    "args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='fanyangxyz33',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size

        if args.use_mnist:
            train_dataset = datasets.MNIST(
                root='./mnist_train',
                train=True,
                download=True,
                transform=script_utils.get_transform_mnist(),
            )

            test_dataset = datasets.MNIST(
                root='./mnist_test',
                train=False,
                download=True,
                transform=script_utils.get_transform_mnist(),
            )

        if args.use_cifar:
            train_dataset = datasets.CIFAR10(
                root='./cifar_train',
                train=True,
                download=True,
                transform=script_utils.get_transform_cifar(),
            )

            test_dataset = datasets.CIFAR10(
                root='./cifar_test',
                train=False,
                download=True,
                transform=script_utils.get_transform_cifar(),
            )

        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        ))
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, drop_last=True, num_workers=8)

        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            print(f'iteration {iteration}')
            diffusion.train()

            x, y = next(train_loader)

            x = x.to(device)
            y = y.to(device)

            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()

            if iteration % args.log_rate == 0:
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        print('testing...')
                        x = x.to(device)
                        y = y.to(device)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        test_loss += loss.item()
                        if args.debug:
                            break

                print('sampling...')
                if args.use_labels:
                    samples = diffusion.sample(
                        args.sample_batch_size, device, y=torch.arange(10, device=device), use_ddim=args.use_ddim)
                else:
                    samples = diffusion.sample(
                        args.sample_batch_size, device, use_ddim=args.use_ddim)

                if args.use_mnist:
                    inv_transform = script_utils.inv_transform_mnist

                if args.use_cifar:
                    inv_transform = script_utils.inv_transform_cifar

                samples = inv_transform(samples).clip(
                    0, 1).permute(0, 2, 3, 1).numpy()

                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                if args.log_to_wandb:
                    wandb.log({
                        "test_loss": test_loss,
                        "train_loss": acc_train_loss,
                        "samples": [wandb.Image(sample) for sample in samples],
                    })

                acc_train_loss = 0

            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)

        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        use_mnist=True,
        use_cifar=False,
        img_channels=1,
        initial_pad=2,
        img_size=28,

        learning_rate=2e-4,
        batch_size=128,
        sample_batch_size=10,
        iterations=800000,

        log_to_wandb=True,
        log_rate=1000,
        checkpoint_rate=1000,
        log_dir="ddpm_logs",
        project_name='diffusion-models-mnist',
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
        debug=False,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
