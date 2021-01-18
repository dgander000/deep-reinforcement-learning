import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from dqn_agent import DQNAgent


def print_env_info(env_info, brain):
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def main():
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)
    print_env_info(env_info, brain)

    n_episodes = 500
    best_score = -np.inf
    scores = []
    eps_history = []
                 
    gamma = 0.99
    lr = 0.0001
    epsilon_start = 1
    epsilon_min = 0.1
    epsilon_decay = 0.99 
    buffer_size = 50000
    batch_size = 32
    update_frequency = 1000
    soft_update = False
    tau = 0.001
    checkpoint_dir='models/'
    agent = DQNAgent(gamma, lr, epsilon_start, epsilon_min, epsilon_decay, state_size, action_size,
                    buffer_size, batch_size, update_frequency, soft_update, tau, checkpoint_dir)

    for i in range(n_episodes):
        score = 0
        done = False
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        while not done:
            action = agent.choose_action(state)
            env_info = env.step(action)[brain_name]       
            next_state = env_info.vector_observations[0]  
            reward = env_info.rewards[0]  
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

        scores.append(score)
        ave_score = np.mean(scores[-100:])
        eps_history.append(agent.epsilon)
        agent.decrement_epsilon()
        if ave_score > best_score:
            best_score = ave_score
        print('episode: ', i,'score: ', score, ' average score %.1f' % ave_score, 
              'best score %.2f' % best_score, 'epsilon %.2f' % agent.epsilon)
        if ave_score >= 13.0:
            print('Environment solved!')
            agent.save_models()
            break

    env.close()
    plot_scores(scores)


if __name__ == '__main__':
    main()