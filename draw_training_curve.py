from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import parsers

## define color
green = (63/225, 135/225, 118/225)


def read_tensorboard_data(tensorboard_path):
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val_loss = ea.scalars.Items("lm loss")
    val_lr = ea.scalars.Items("learning-rate")
    val_grad = ea.scalars.Items("grad-norm")
    return val_loss, val_lr, val_grad
 
def draw_plt(val_loss, val_lr, val_grad):
    # plt.figure()
    fig, axs = plt.subplots(1,2)
    ax2 = axs[0].twinx()

    axs[0].plot([i.step for i in val_loss], [j.value for j in val_loss], label="lm loss")
    axs[0].set_xlabel('Iteration', fontsize=14)
    axs[0].set_ylabel("Lm Loss", fontsize=14)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_title('LM Loss', fontsize=18)

    axs[0].grid(True)
    
    ax2.plot([i.step for i in val_lr], [j.value for j in val_lr], label="learning-rate", color=green)
    ax2.tick_params(axis='y', colors=green)
    ax2.set_ylabel('Learning Rate', fontsize=14 , color=green)
    ax2.spines['right'].set_color(green)

    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    axs[1].plot([i.step for i in val_grad], [j.value for j in val_grad], label="grad-norm")
    axs[1].set_xlabel('Iteration', fontsize=14)
    axs[1].set_ylabel("Grad Norm", fontsize=14)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_title('Grad Norm', fontsize=18)

    axs[1].grid(True)
    plt.savefig("./output/traing_curve.png")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorboard_path", default='./logs/events.out.tfevents.1692534714.e10r2n12.1701.0', type=str)
    args = parser.parse_args()

    val_loss, val_lr, val_grad = read_tensorboard_data(args.tensorboard_path)
    draw_plt(val_loss, val_lr, val_grad)
