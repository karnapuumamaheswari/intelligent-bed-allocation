from evaluation.evaluate import compare_policies
from training.train_dqn import train


if __name__ == "__main__":
    print("Training DQN model...")
    train()

    print("\nEvaluating trained policy against the rule-based baseline...")
    compare_policies()
