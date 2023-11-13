from sde_gan_OU import Generator, Discriminator, get_data

import fire
import matplotlib.pyplot as plt

import torch
import torch.optim.swa_utils as swa_utils
import torchcde
import torchsde


def plot(ts, generator, dataloader, num_plot_samples, plot_locs, zone):
    # Get samples
    real_samples, = next(iter(dataloader))
    assert num_plot_samples <= real_samples.size(0)
    real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
    real_samples = real_samples[..., 1]

    with torch.no_grad():
        generated_samples = generator(ts, real_samples.size(0)).cpu()
    generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)
    generated_samples = generated_samples[..., 1]

    # Plot histograms
    for prop in plot_locs:
        time = int(prop * (real_samples.size(1) - 1))
        real_samples_time = real_samples[:, time]
        generated_samples_time = generated_samples[:, time]
        
        plt.figure()
        _, bins, _ = plt.hist(real_samples_time.cpu().numpy(), bins=32, alpha=0.7, label='Real', color='dodgerblue',
                              density=True)
        bin_width = bins[1] - bins[0]
        num_bins = int((generated_samples_time.max() - generated_samples_time.min()).item() // bin_width)

        plt.hist(generated_samples_time.cpu().numpy(), bins=num_bins, alpha=0.7, label='Generated', color='crimson',
                 density=True)
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Marginal distribution at time {time}.')
        plt.tight_layout()
        plt.savefig(f'./torchsde/images/{zone}/marginal_distribution_{prop}_{zone}.png', dpi=200, format='png')
        #plt.show()

    real_samples = real_samples[:num_plot_samples]
    generated_samples = generated_samples[:num_plot_samples]

    # Plot samples
    plt.figure()
    real_first = True
    generated_first = True
    for real_sample_ in real_samples:
        kwargs = {'label': 'Real'} if real_first else {}
        plt.plot(ts.cpu(), real_sample_.cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
        real_first = False
    for generated_sample_ in generated_samples:
        kwargs = {'label': 'Generated'} if generated_first else {}
        plt.plot(ts.cpu(), generated_sample_.cpu(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
        generated_first = False
    plt.legend()
    plt.title(f"{num_plot_samples} samples from both real and generated distributions.")
    plt.tight_layout()
    plt.savefig(f'./torchsde/images/{zone}/samples_real_vs_generated_{zone}.png', dpi=200, format='png')
    #plt.show()


def main(
    zone='OU',
    
    # Architectural hyperparameters. These are quite small for illustrative purposes.
    initial_noise_size=5,  # How many noise dimensions to sample at the start of the SDE.
    noise_size=3,          # How many dimensions the Brownian motion has.
    hidden_size=16,        # How big the hidden size of the generator SDE and the discriminator CDE are.
    mlp_size=16,           # How big the layers in the various MLPs are.
    num_layers=1,          # How many hidden layers to have in the various MLPs.

    # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
    batch_size=612,         # Batch size.

    # Evaluation and plotting hyperparameters
    steps_per_print=10,                   # How often to print the loss.
    num_plot_samples=50,                  # How many samples to use on the plots at the end.
    plot_locs=(0.1, 0.3, 0.5, 0.7, 0.9),  # Plot some marginal distributions at this proportion of the way along.
):
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")

    # Data
    ts, data_size, train_dataloader = get_data(zone=zone, batch_size=batch_size, device=device)
    # infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

    # Models
    generator = Generator(data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers).to(device)
    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers).to(device)
    
    generator.load_state_dict(torch.load(f'./trained_model/generator_{zone}', map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load(f'./trained_model/discriminator_{zone}', map_location=torch.device('cpu')))

    _, _, test_dataloader = get_data(zone=zone, batch_size=batch_size, device=device)
    
    # print('Saving model...')
    # if not os.path.isdir('./trained_model'):
    #     os.makedirs('./trained_model')
    # torch.save(generator.state_dict(), f'./WIND/trained_model/generator_{zone}')
    # torch.save(discriminator.state_dict(), f'./WIND/trained_model/discriminator_{zone}')

    plot(ts, generator, test_dataloader, num_plot_samples, plot_locs, zone)


if __name__ == '__main__':
    fire.Fire(main)
    