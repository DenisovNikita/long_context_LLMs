import sys
sys.path.append("..")
from definitions import *


def plot_tokens_distributions(tokens_dir_path: Path):
    values = []
    labels = []
    for file in tokens_dir_path.glob("*"):
        labels.append(file.stem)
        with open(file, "r") as f:
            values.append(list(json.load(f).values()))
    values = np.array(values)
    labels = np.array(labels)
    perm = range(len(values))
    perm = sorted(perm, key=lambda i: -np.mean(values[i]))
    perm = np.array(perm)
    fig, ax = plt.subplots()
    for position, column in enumerate(perm):
        ax.boxplot(values[column], positions=[position], vert=False, widths=0.5)
    ax.set_yticks(range(position+1))
    ax.set_yticklabels(labels[perm])
    ax.vlines(np.array([0, 0.5, 1]) * 1e6, ymin=-1, ymax=len(labels), alpha=0.2, color='blue')
    ax.vlines([128000], ymin=-1, ymax=len(labels), alpha=0.8, color='red')
    plt.xlabel("Количество токенов")
    plt.title(f"Распределения токенов по {len(values[0])} {tokens_dir_path.stem} для разных моделей")

    plt.show()


def plot_tokens_distributions_by_attempt(tokens_path: Path):
    with open(tokens_path, "r") as f:
        tokens = json.load(f)["tokens"]
    n_values = list(tokens.keys())
    values = []
    for n in n_values:
        values.append(tokens[n])
    values = np.array(values)
    labels = np.array(n_values)
    perm = range(len(values))
    perm = sorted(perm, key=lambda i: -np.mean(values[i]))
    perm = np.array(perm)
    fig, ax = plt.subplots()
    for position, column in enumerate(perm):
        ax.boxplot(values[column], positions=[position], vert=False, widths=0.5)
    ax.set_yticks(range(position+1))
    ax.set_yticklabels(labels[perm])
    ax.vlines([4096], ymin=-1, ymax=len(labels), alpha=0.2, color='blue')
    # ax.vlines([128000], ymin=-1, ymax=len(labels), alpha=0.8, color='red')
    plt.xlabel("Количество токенов")
    plt.ylabel("n")
    plt.title(f"Распределения токенов по {len(values[0])} попыткам для разных n")

    plt.show()
        
