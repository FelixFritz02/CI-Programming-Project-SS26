import matplotlib.pyplot as plt
import numpy as np


def plot_training_progress(reward_history, monotonicity_history, title="Training Progress"):
    """
    Plottet kumulierten Reward (linke Achse) zusammen mit den Monotonie-Verhältnissen
    aus evaluate_monotonicity_systematic (rechte Achse, 0-1).

    monotonicity_history: Liste von (episode, mono_c, mono_t, mono_r, mono_q, mono_mixed)
    -Tupeln, wie von DQNAgent.train() zurückgegeben.
    """
    episodes   = [x[0] for x in monotonicity_history]
    mono_c     = [x[1] for x in monotonicity_history]
    mono_t     = [x[2] for x in monotonicity_history]
    mono_r     = [x[3] for x in monotonicity_history]
    mono_q     = [x[4] for x in monotonicity_history]
    mono_mixed = [x[5] for x in monotonicity_history]

    fig, ax1 = plt.subplots()
    ax1.plot(reward_history, color="blue", label="Cumulated Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulated Reward", color="blue")

    ax2 = ax1.twinx()
    ax2.plot(episodes, mono_r,     color="red",    label="Monotonicity reward r (systematic)")
    ax2.plot(episodes, mono_c,     color="green",  label="Monotonicity C_k (systematic)")
    ax2.plot(episodes, mono_t,     color="orange", label="Monotonicity t (systematic)")
    ax2.plot(episodes, mono_q,     color="black",  label="Monotonicity q (systematic)")
    ax2.plot(episodes, mono_mixed, color="purple", label="Monotonicity mixed (systematic)")
    ax2.set_ylabel("Monotonicity Ratio")
    ax2.set_ylim(0, 1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower left", fontsize="small")

    plt.title(title)
    fig.tight_layout()
    plt.show()


def plot_depth_epsilon(depth_history, title="Episodentiefe und Epsilon über das Training"):
    """
    Plottet die erreichte Tiefe (Schritte je Episode, linke Achse) zusammen mit
    Epsilon (rechte Achse).

    depth_history: Liste von (episode, tiefe, epsilon)-Tupeln, wie von
    DQNAgent.train() zurückgegeben.
    """
    episodes = [x[0] for x in depth_history]
    depths   = [x[1] for x in depth_history]
    epsilons = [x[2] for x in depth_history]

    fig, ax1 = plt.subplots()
    ax1.plot(episodes, depths, color="blue", label="Tiefe (Schritte in Episode)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Tiefe (Schritte)", color="blue")

    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilons, color="red", label="Epsilon")
    ax2.set_ylabel("Epsilon", color="red")

    plt.title(title)
    fig.tight_layout()
    plt.show()


def plot_eval_rewards(eval_rewards, window=50, title="Eval-Reward-History", verbose=True):
    """
    Plottet den Reward-Verlauf einer Evaluationsphase (z.B. DQNAgent.evaluate())
    zusammen mit gleitendem Mittelwert und Gesamt-Mittelwert, und gibt
    Kennzahlen (Mean/Std/Median/Max/Min) aus.

    Einheitliche Eval-Darstellung, damit standard_dqn, deep_lattice etc. direkt
    vergleichbar sind (gleiche Kennzahlen, gleiches Plot-Layout).

    Parameter
    ----------
    eval_rewards : Liste/Array kumulierter Rewards je Eval-Episode
    window        : Fenstergröße für gleitenden Mittelwert
    title         : Plot-Titel
    verbose       : Kennzahlen ausgeben

    Rückgabe
    --------
    stats: dict mit n, mean, std, median, max, min
    """
    r = np.array(eval_rewards)
    n = len(r)

    stats = {
        "n": n,
        "mean": r.mean(),
        "std": r.std(ddof=1) if n > 1 else 0.0,
        "median": np.median(r),
        "max": r.max(),
        "min": r.min(),
    }

    if verbose:
        print(f"Eval-Episoden: {stats['n']}")
        print(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Median: {stats['median']:.4f}")
        print(f"Max: {stats['max']:.4f}, Min: {stats['min']:.4f}")

    plt.figure(figsize=(9, 4))
    plt.plot(r, alpha=0.4, label="Cumulated Reward pro Eval-Episode")
    if n >= window:
        mov = np.convolve(r, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window - 1, n), mov, color="orange", label=f"{window}-Episode moving average")
    plt.axhline(stats["mean"], color="red", linestyle="--", label=f"Mean = {stats['mean']:.2f}")
    plt.xlabel("Eval-Episode")
    plt.ylabel("Cumulated Reward")
    plt.title(title)
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.show()

    return stats


def plot_reward_moving_average(reward_history, window=50, title="Reward History", figsize=(9,4), verbose=True):
    """Plot kumulierter Reward pro Episode mit gleitendem Mittelwert.

    Parameters
    ----------
    reward_history : list/array
        Kumulierte Rewards pro Trainings-Episode
    window : int
        Fenstergröße für gleitenden Mittelwert
    title : str
        Plot-Titel
    figsize : tuple
        Figurgröße
    verbose : bool
        Kennzahlen (mean/std/median/max/min) ausgeben

    Returns
    -------
    dict
        Statistische Kennzahlen
    """
    r = np.array(reward_history)
    n = len(r)

    stats = {
        "n": n,
        "mean": r.mean() if n>0 else float('nan'),
        "std": r.std(ddof=1) if n > 1 else 0.0,
        "median": np.median(r) if n>0 else float('nan'),
        "max": r.max() if n>0 else float('nan'),
        "min": r.min() if n>0 else float('nan'),
    }

    if verbose:
        print(f"Episoden: {stats['n']}")
        print(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Median: {stats['median']:.4f}")
        print(f"Max: {stats['max']:.4f}, Min: {stats['min']:.4f}")

    plt.figure(figsize=figsize)
    plt.plot(r, label='Cumulated Reward per Episode', alpha=0.6)
    if n >= window and window > 0:
        mov = np.convolve(r, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, n), mov, color='red', label=f'{window}-Episode moving average')
    plt.xlabel('Episode')
    plt.ylabel('Cumulated Reward')
    plt.title(title)
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.show()

    return stats


def plot_training_diagnostics(reward_history, depth_history=None, monotonicity_history=None, window=50, title="Training Diagnostics"):
    """
    Erweiterte Diagnose-Plots:
      - Reward + gleitender Mittelwert, erkennt plötzlichen Sprung
      - Epsilon-Verlauf (falls `depth_history` gegeben)
      - Episodentiefe (falls `depth_history` gegeben)
      - Monotonie-Ratios (falls `monotonicity_history` gegeben)

    Parameter:
      reward_history          : Liste oder Array kumulierter Rewards pro Episode
      depth_history           : Liste von (episode, depth, epsilon)-Tupeln (optional)
      monotonicity_history    : Liste von (episode, mono_c, mono_t, mono_r, mono_q, mono_mixed)-Tupeln (optional)
      window                  : Fenstergröße für gleitenden Mittelwert
    """
    r = np.array(reward_history)
    n = len(r)

    # Gleitender Durchschnitt (valid -> length n-window+1)
    if n >= 2 and window >= 1:
        mov = np.convolve(r, np.ones(window) / window, mode='valid')
    else:
        mov = None

    # Sprungerkennung: erster Index, bei dem mov > mean + 2*std
    jump_idx = None
    if mov is not None and len(mov) > 0:
        thr = mov.mean() + 2.0 * mov.std()
        idxs = np.where(mov > thr)[0]
        if len(idxs) > 0:
            # convert mov-index to episode index (mov index i corresponds to episode i+window-1)
            jump_idx = int(idxs[0] + window - 1)

    # Layout: 2x2, einige Panels optional
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Reward + mov
    ax_r = axes[0, 0]
    ax_r.plot(r, label='Cumulated Reward', alpha=0.6)
    if mov is not None:
        ax_r.plot(np.arange(window - 1, n), mov, label=f'{window}-ep moving avg', color='red')
    if jump_idx is not None:
        ax_r.axvline(jump_idx, color='magenta', linestyle='--', label=f'Jump ~ep {jump_idx}')
    ax_r.set_xlabel('Episode')
    ax_r.set_ylabel('Cumulated Reward')
    ax_r.legend(fontsize='small')
    ax_r.grid(True)

    # Monotonicity (optional)
    ax_m = axes[0, 1]
    if monotonicity_history is not None and len(monotonicity_history) > 0:
        episodes = [x[0] for x in monotonicity_history]
        mono_c     = [x[1] for x in monotonicity_history]
        mono_t     = [x[2] for x in monotonicity_history]
        mono_r     = [x[3] for x in monotonicity_history]
        mono_q     = [x[4] for x in monotonicity_history]
        mono_mixed = [x[5] for x in monotonicity_history]

        ax_m.plot(episodes, mono_r,     color='red',    label='mono r')
        ax_m.plot(episodes, mono_c,     color='green',  label='mono C')
        ax_m.plot(episodes, mono_t,     color='orange', label='mono t')
        ax_m.plot(episodes, mono_q,     color='black',  label='mono q')
        ax_m.plot(episodes, mono_mixed, color='purple', label='mono mixed')
        ax_m.set_ylim(0, 1)
        ax_m.set_xlabel('Episode')
        ax_m.set_ylabel('Monotonicity Ratio')
        ax_m.legend(fontsize='small')
        ax_m.grid(True)
    else:
        ax_m.text(0.5, 0.5, 'No monotonicity data', ha='center', va='center')
        ax_m.axis('off')

    # Epsilon (from depth_history)
    ax_e = axes[1, 0]
    if depth_history is not None and len(depth_history) > 0:
        eps = [x[2] for x in depth_history]
        ax_e.plot(range(1, len(eps) + 1), eps, color='tab:red')
        ax_e.set_xlabel('Episode')
        ax_e.set_ylabel('Epsilon')
        ax_e.grid(True)
    else:
        ax_e.text(0.5, 0.5, 'No depth/epsilon data', ha='center', va='center')
        ax_e.axis('off')

    # Depth
    ax_d = axes[1, 1]
    if depth_history is not None and len(depth_history) > 0:
        depths = [x[1] for x in depth_history]
        ax_d.plot(range(1, len(depths) + 1), depths, color='tab:blue')
        ax_d.set_xlabel('Episode')
        ax_d.set_ylabel('Episode depth (steps)')
        ax_d.grid(True)
    else:
        ax_d.text(0.5, 0.5, 'No depth data', ha='center', va='center')
        ax_d.axis('off')

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
