def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    T = len(rewards)

    G = [0.0]*T
    G[-1] = rewards[-1]
    for t in range(T-2, -1, -1):
        G[t] = rewards[t] + gamma * G[t+1]

    MeanG = sum(G)/T

    A = [g - MeanG for g in G]

    loss = - sum(lp * a for lp, a in zip(log_probs, A)) / T

    return loss