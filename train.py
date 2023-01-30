from ddpg import DDPG as Agent
from noiseObjects import *
import matplotlib.pyplot as plt
import gym
     
def _train_on_env(env, N_training = 10, noiseObject = None, verbose=False):
    
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.shape[0] 
    up_bounds = env.action_space.high
    low_bounds = env.action_space.low

    agent = Agent(state_size=state_size, action_size=action_size,
                  lower_bounds=low_bounds, upper_bounds=up_bounds,
                  verbose=verbose)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Maximum number of steps
    max_steps = env._max_episode_steps
    
    if verbose :
        print("="*70)
        print("Training on ",N_training," epochs")
        print("Max steps : ",max_steps)

    for ep in range(N_training):
        
        state, info = env.reset() #tuple (np.array, dict)
        episodic_reward = 0
        noise = None
        for _ in range(max_steps):
            
            # add noise if given:
            if noiseObject:
                noise = noiseObject()
            action = agent.policy(state,noise=noise)

            observation, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, observation, done)    
            agent.learn()
            
            episodic_reward += reward

            # End this episode when `done` is True
            if done : 
                break

            state = observation
        #self.noise.reset()
        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        if verbose:
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
    
    # Episodes versus Avg. Rewards
    fig = plt.gcf()
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
    plt.grid()
    plt.draw()

    print("\n")

    saving = input("saving training session ? [y/n]")
    if saving == 'y':
        filename = input("input filename : ")
        print("saving models under ", filename)
        agent.save_models(filename)
        fig.savefig(filename + "/train_avg_reward.png")

def main():

    ### GYM ENVIRONMENT FOR TESTING ###

    render_mode = "human"

    ### PENDULUM-V1#
    #problem = "Pendulum-v1"
    #env = gym.make(problem,render_mode=render_mode)
    #N_training = 30

    ### LUNAS-LANDER-V2 ###
    problem = "LunarLander-v2"
    env = gym.make(problem, continuous = True, gravity = -10.0, enable_wind = False, wind_power = 15.0, turbulence_power = 1.5,render_mode=render_mode)
    N_training = 100

    ### CARRACING-V1 ###
    #problem = "CarRacing-v2"
    #env = gym.make(problem, domain_randomize=True,render_mode=render_mode)

    ### HALFCHEETAH-V2 ###
    #https://mujoco.org/
    #problem = 'HalfCheetah-v2'
    #env = gym.make(problem,render_mode=render_mode)

    #### NOISE OBJECT ####
    #noise from Ornstein-Uhlenbeck process for action exploration
    #https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    action_size = env.action_space.shape[0]
    mean=np.zeros(action_size)
    std_dev = 0.02 * np.ones(action_size)
    theta   = 0.15
    dt      = 1e-2 
    noiseObject = OUNoise(mean=mean,
                          std_deviation=std_dev,
                          theta=theta,
                          dt=dt,
                          x_initial=None)

    _train_on_env(env,N_training = N_training,
                   noiseObject=None,
                   verbose=True)

if __name__ == "__main__":
    main()
