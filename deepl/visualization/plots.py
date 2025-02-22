import matplotlib.pyplot as plt


# Plot the loss values over iterations
def loss_plot(trainer, point_size, loss_str):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(trainer.loss_values)),
                trainer.loss_values,
                label='Loss',
                s=len(trainer.loss_values)*[point_size])
    plt.xlabel('Iterations')
    plt.ylabel(f'{loss_str}')
    plt.title(f'{loss_str} vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot the mean loss validation values over iterations
def mean_validation_plot(trainer, point_size, loss_str):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(trainer.mean_validation_loss)),
                trainer.mean_validation_loss,
                label='Loss',
                s=len(trainer.mean_validation_loss)*[point_size])
    plt.xlabel('Iterations')
    plt.ylabel(f'Mean validation {loss_str}')
    plt.title(f'Mean validation {loss_str} vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.show()
