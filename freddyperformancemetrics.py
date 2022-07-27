# quick functions for agent evaluation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def jumpstart(y):
    return (y[0])


def asymptotic_performance(y, rolling_avg=1):
    asymptotic_performance = np.mean(y[-rolling_avg:])
    return asymptotic_performance


def total_reward_fnc(y):
    return np.sum(y)


def transfer_ratio(y_transfer, y_reference, total_reward):
    transfer_ratio = total_reward(y_transfer) / total_reward(y_reference)
    return transfer_ratio


def time_to_threshold(x, y, threshold=250):
    return x[np.argmax(y > threshold)]


def max_reward_fnc(y):
    return np.max(y), np.argmax(y)


def performance_metrics(x, y, rolling_avg=10, threshold=275, plot=True):
    '''
    TODO: write plots functions
    '''

    max_reward, idx_training_max = max_reward_fnc(y)
    total_reward = total_reward_fnc(y)
    n_timesteps = x.max()
    reward_per_timestep = total_reward / n_timesteps
    asymptote = asymptotic_performance(y, rolling_avg=rolling_avg)
    std_asymptote = np.std(y[-rolling_avg:])
    asymptote_after_max = asymptotic_performance(y, rolling_avg=n_timesteps - idx_training_max)
    std_asymptote_after_max = np.std(y[-(n_timesteps - idx_training_max):])
    time_threshold = time_to_threshold(x, y, threshold=threshold)
    threshold_80 = 0.80 * threshold
    time_threshold_80_of_max = time_to_threshold(x, y, threshold=threshold_80)
    print(0.80 * max_reward)

    print(f" Max reward: {max_reward} \
        \n Time steps to max reward: {idx_training_max}/{n_timesteps} ({idx_training_max * 100 / n_timesteps:.2f}% of training period) \
        \n Total reward: {total_reward:.2f} \
        \n Reward per timestep: {reward_per_timestep:.2f} \
        \n Asymptotic performance (last {rolling_avg} timesteps): {asymptote:.2f} ±{std_asymptote:.2f} ({asymptote * 100 / max_reward:.2f}% of max reward) \
        \n Asymptotic performance after max: {asymptote_after_max:.2f} ±{std_asymptote_after_max:.2f} ({asymptote_after_max * 100 / max_reward:.2f}% of max reward) \
        \n Time to threshold(={threshold}): {time_threshold}/{n_timesteps} ({time_threshold * 100 / n_timesteps:.2f}% of training period) \
        \n Time to threshold(80% of max={threshold_80}): {time_threshold_80_of_max}/{n_timesteps} ({time_threshold_80_of_max * 100 / n_timesteps:.2f}% of training period)" \
          )

    if plot == True:
        sns.scatterplot(x=x, y=y, color='green')
        plt.title('Reward Over Timesteps 2')
        plt.xlabel('Episode Index')
        plt.ylabel('Reward')
        plt.show()

        # sns.barplot(data=[asymptote])
        # plt.show()
        numpy_data = ([threshold, time_threshold_80_of_max])

        plt.bar([0, 1], [time_threshold, time_threshold_80_of_max])
        plt.xticks([0, 1], labels=['Time to Max Threshold', 'Time to 80% max threshold'])
        plt.title('Threshold Time')
        plt.show()
