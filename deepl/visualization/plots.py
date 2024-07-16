import matplotlib.pyplot as plt

def loss_plot(trainer, point_size, loss_str):
    # Plot the loss values over iterations
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(trainer.loss_values)),
                trainer.loss_values,
                label='Loss',
                s=len(trainer.loss_values)*[point_size])
    plt.xlabel(f'Iterations')
    plt.ylabel(f'{loss_str}')
    plt.title(f'{loss_str} vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.show()
