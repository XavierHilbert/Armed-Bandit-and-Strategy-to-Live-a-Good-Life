import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# Defining a Bandit
class Bandit:
    def __init__(self, true_mean, cost, standard_deviation):
        self.true_mean = true_mean
        self.cost = cost
        self.standard_deviation = standard_deviation
        self.true_efficiency = self.true_mean / self.cost
        self.calculated_mean = 0
        self.number_of_hits = 1e-6

    @property
    def calculated_efficiency(self):
        return self.calculated_mean / self.cost

    def hit(self):
        return self.true_mean + np.random.randn()*self.standard_deviation

    def update(self, x):
        self.number_of_hits += 1
        self.calculated_mean = (1 - 1.0/self.number_of_hits)*self.calculated_mean + 1.0/self.number_of_hits*x


# Defining strategies
def epsilon_greedy(_, bandits, epsilon):
    p = np.random.random()
    if p < epsilon:
        return np.random.choice(len(bandits))
    else:
        return np.argmax([b.calculated_efficiency for b in bandits])

def ucb1(step, bandits):
    return np.argmax([b.calculated_efficiency + np.sqrt(2*np.log(step+1)/(b.number_of_hits)) for b in bandits])


def run_experiment(bandits, N, strategy):
    # creating bandits and setting up data to collect
    bandits = [Bandit(*b) for b in bandits]
    data = np.zeros(N)
    step = 0

    # running the experiment
    while step < N:
        choice_index = strategy(step, bandits)
        x = bandits[choice_index].hit()
        bandits[choice_index].update(x)
        # recording reward for the plot
        data[step] = x
        
        # reducing the number of pulls available by the cost of the bandit
        step += bandits[choice_index].cost

    cumulative_average = np.cumsum(data)/(np.arange(N)+1)

    return cumulative_average

def run(bandits, N, number_of_trials, title = None):
    epsilon1_greedy_results = np.array([run_experiment(bandits, N, partial(epsilon_greedy, epsilon = .1)) for _ in range(number_of_trials)]) # most greedy
    epsilon1_greedy_results = epsilon1_greedy_results.mean(axis=0)

    epsilon2_greedy_results = np.array([run_experiment(bandits, N, partial(epsilon_greedy, epsilon = .2)) for _ in range(number_of_trials)])
    epsilon2_greedy_results = epsilon2_greedy_results.mean(axis=0)

    epsilon3_greedy_results = np.array([run_experiment(bandits, N, partial(epsilon_greedy, epsilon = .3)) for _ in range(number_of_trials)]) # least greedy
    epsilon3_greedy_results = epsilon3_greedy_results.mean(axis=0)

    ucb1_results = np.array([run_experiment(bandits, N, ucb1) for _ in range(number_of_trials)])
    ucb1_results = ucb1_results.mean(axis=0)

    optimal_bandit = max(bandits, key=lambda x: x[0]/x[1])
    optimal_bandit_results = np.ones(N) * optimal_bandit[0]/optimal_bandit[1]  # optimal_bandit[1] = cost, optimal_bandit[0] = true_mean

    if title is not None: plt.title(title)
    plt.plot(optimal_bandit_results, label='Optimal (Assuming True Efficiency Every Pull)')
    plt.plot(epsilon1_greedy_results, label='Epsilon Greedy, Epsilon = 0.1')
    plt.plot(epsilon2_greedy_results, label='Epsilon Greedy, Epsilon = 0.2')
    plt.plot(epsilon3_greedy_results, label='Epsilon Greedy, Epsilon = 0.3')
    plt.plot(ucb1_results, label='UCB1')
    plt.legend()
    plt.ylabel("Average Utility")
    plt.xlabel("Number of Pulls")
    plt.show()


def createBandits(number_of_bandits, means, costs, standard_deviations):
    # create a list of bandits
    bandits = []
    for _ in range(number_of_bandits):
        bandits.append([np.random.choice(means), np.random.choice(costs), np.random.choice(standard_deviations)])
    return bandits


if __name__ == '__main__':
    """3 bandits: true mean, cost, standard deviation. Where reward = N(mean, standard deviation). 
    Uncomment to use pre-defined bandits"""
    """BANDITS = [
            [2, 1,1], 
            [3, 2,1],
            [4, 3,1],
            [6,3,1],
            [10,7,1],
            [13,15,1]
        ]"""
   
    
    """Constructing bandits at random. Where reward = N(mean, standard deviation) with a random cost associated with each bandit.
    Constructed by picking a random number from first array as mean, second array as cost, and third array as standard deviation"""
    NUMBER_OF_BANDITS = 100
    BANDITS = createBandits(NUMBER_OF_BANDITS, [1,2,3,4,5,6,6,7,8,9,11,12,13,14,15,16, 20], [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9,10,11,12])


    """Running experiment once."""
    NUMBER_OF_PULLS = 10000
    run(BANDITS, NUMBER_OF_PULLS, 1)
   
    """Getting average behavior over many trials. May take a while."""
    NUMBER_OF_TRIALS = 10

    # Assumes relatively small Budget.
    NUMBER_OF_PULLS = 20
    run(BANDITS, NUMBER_OF_PULLS, NUMBER_OF_TRIALS, f"{NUMBER_OF_PULLS} pulls with {len(BANDITS)} Bandits Averaged Amongst {NUMBER_OF_TRIALS} trials")

    # Assumes relatively large Budget.
    NUMBER_OF_PULLS = 10000
    run(BANDITS, NUMBER_OF_PULLS, NUMBER_OF_TRIALS, f"{NUMBER_OF_PULLS} pulls with {len(BANDITS)} Bandits Averaged Amongst {NUMBER_OF_TRIALS} trials")





