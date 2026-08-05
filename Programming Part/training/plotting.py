import matplotlib.pyplot as plt


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
