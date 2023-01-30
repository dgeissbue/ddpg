#Deep Deterministic Policy Gradient (DDPG)
# https://arxiv.org/abs/1509.02971
# https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
# code from the excellent article :
# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
# modificitation to take into account continuous state and/or action:
# https://keras.io/examples/rl/ddpg_pendulum/
# Structure of the class widely inspired by :
# https://github.com/cookbenjamin/DDPG

import numpy as np
import tensorflow as tf
from keras.initializers import normal, identity
from keras.models import model_from_json, Sequential, Model
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda
from keras.optimizers import Adam

#Deep Deterministic Policy Gradient (DDPG)
class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG) implementation
    """

    def __init__(self, state_size, action_size, lower_bounds, upper_bounds,
                 agent_hidden_units = [(256,"relu"),(256,"relu")],
                 critic_hidden_units = [[(16,"relu"),(32,"relu")],[(32,"relu")],[(256,"relu"),(256,"relu")]],
                 agent_learning_rate = 0.001, critic_learning_rate = 0.002,
                 gamma = 0.99, tau = 0.005, batch_size = 64,
                 memory_limit = 5000, verbose=False):
        """
        Constructor for the Deep Deterministic Policy Gradient object adapted 
        for continuous action

        :param state_size: An integer denoting the dimensionality of the states
            in the current problem
        :param action_size: An integer denoting the dimensionality of the
            actions in the current problem
       """

        # Store parameters
        self.state_size  = state_size
        self.action_size = action_size 
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.agent_HU = agent_hidden_units
        self.critic_HU = critic_hidden_units
        self.batch_size = batch_size 
        self.memory_limit = memory_limit
        self.gamma = gamma
        self.tau = tau
        self.agent_LR = agent_learning_rate  
        self.critic_LR= critic_learning_rate 

        # Print informations
        if verbose:
            print("Size of State Space ->  {}".format(self.state_size))
            print("Size of Action Space ->  {}".format(self.action_size))
            print("Max Value of Action ->  {}".format(self.upper_bounds))
            print("Min Value of Action ->  {}".format(self.lower_bounds))

        # Prepare the buffer 
        self.buffer_counter = 0 #count memory slots used
        self.state_buffer = np.zeros((self.memory_limit, self.state_size ))
        self.action_buffer = np.zeros((self.memory_limit, self.action_size))
        self.reward_buffer = np.zeros((self.memory_limit,1))
        self.observation_buffer = np.zeros((self.memory_limit, self.state_size ))
        self.continue_buffer = np.zeros((self.memory_limit,1))
        
        # Optimizer for learning
        self.agent_optimizer = Adam(self.agent_LR)
        self.critic_optimizer = Adam(self.critic_LR)

        # Generate the agent and the critic models
        self.agent  = self.create_agent(verbose)
        self.critic = self.create_critic(verbose)
        
        # "hack" implemented by DeepMind to improve convergence
        # Generate carbon copy of the model so that we avoid divergence
        self.target_agent  = self.create_agent()
        self.target_critic = self.create_critic()
        # Making the weights equal initially
        self.target_agent.set_weights(self.agent.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def _create_layers(self,layers,inputs):
        """
        Helper function for piling up NN layers
        :param layers: list of tuple (size, activation) with size an integer
        denoting the size of the layer and activation a string denoting the 
        activation function of the layer
        :param inputs: reference on the inputs layer
        :return: reference on the last NN layer
        """
        for depth,layer in enumerate(layers):
            size = layer[0]
            activation = layer[1]
            if depth == 0:
                out = Dense(size, activation=activation)(inputs)
            else :
                out = Dense(size, activation=activation)(out)
        return out

    def create_agent(self,verbose=False):
        """
        Generates the agent model based on the hyperparameters defined in the
        constructor.
        :return: reference to the model
        """
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        inputs  = Input(shape=(self.state_size,))
        out = self._create_layers(self.agent_HU,inputs)
        outputs = Dense(self.action_size, activation="tanh", kernel_initializer=last_init)(out)

        # Hyperbolic tangent output should fall within action bounds
        mean = (self.upper_bounds + self.lower_bounds)/2
        dev  = (self.upper_bounds - self.lower_bounds)/2
        outputs = outputs * dev + mean

        model   = Model(inputs, outputs)
        model.compile()
        if verbose : print(model.summary())

        return model

    def create_critic(self, verbose=False):
        """
        Generates the critic model based on the hyperparameters defined in the
        constructor.
        :return: reference to the model
        """
        state_layers  = self.critic_HU[0]
        action_layers = self.critic_HU[1]
        concat_layers = self.critic_HU[2]

        # State as input
        state_inputs = Input(shape=(self.state_size,))
        state_out = self._create_layers(state_layers,state_inputs)

        # Action as input
        action_inputs = Input(shape=(self.action_size,))
        action_out = self._create_layers(action_layers,action_inputs)

        # Both are passed through seperate layer before concatenating
        concat = Concatenate()([state_out, action_out])
        out = self._create_layers(concat_layers,concat)
        outputs = tf.keras.layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_inputs, action_inputs], outputs)
        model.compile()
        if verbose : print(model.summary())

        return model

    def policy(self, state, noise=None):
        """
        Returns the action predicted by the agent given the current state.
        Noise can be added for action exploration
        :param state: numpy array denoting the current state.
        :return: numpy array denoting the predicted action.
        """
        # Convert state to Tensor
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sampled_actions = tf.squeeze(self.agent(tf_state))

        # Adding noise to action
        if np.array(noise != None).all() :
            sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bounds, self.upper_bounds)
        action = np.squeeze(legal_action)
        
        # Deal with size 1 actions :
        if action.size == 1: 
            return [action]
        else:
            return action

    def remember(self, state, action, reward, observation, done):
        """
        Stores the given state, action, reward etc in the Agent's memory.
        :param state: The state to remember
        :param action: The action to remember
        :param reward: The reward to remember
        :param observation: The state after the action (if applicable)
        :param done: Whether this was a final state
        :return: None
        """

        # Set index to zero if memory limit achieved
        index = self.buffer_counter % self.memory_limit

        self.state_buffer[index]       = state
        self.action_buffer[index]      = action
        self.reward_buffer[index]      = reward
        self.observation_buffer[index] = observation
        self.continue_buffer[index]  = not done

        self.buffer_counter += 1

    def learn(self):
        """
        Finds a random sample of size self.batch_size from the agent's current
        memory and call training upon the batch
        :return: None
        """
        # Get sampling range
        record_range = min(self.buffer_counter, self.memory_limit)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        
        # Convert the batch to tensors for NN evaluation
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        observation_batch = tf.convert_to_tensor(self.observation_buffer[batch_indices])
        continue_batch = tf.convert_to_tensor(self.continue_buffer[batch_indices])
        continue_batch = tf.cast(continue_batch, dtype=tf.float32)
        
        # Train on the batch
        self.update(state_batch, action_batch, reward_batch, observation_batch, continue_batch)

    def _get_q_targets(self, observations, continues, rewards):
        """
        Calculates the q targets with the following formula
        q = r + gamma * next_q
        unless there is no next state in which
        q = r
        :param observations: List(List(Float)) Denoting the t+1 state
        :param continues: List(Bool) denoting whether each step was an exit step
        :param rewards: List(Float) Denoting the reward given in each step
        :return: The q targets
        """
        next_actions  = self.target_agent(observations, training=True)
        next_q_values = self.target_critic([observations,next_actions])

        q_targets = rewards
        q_targets += continues * next_q_values* self.gamma

        return q_targets

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, observation_batch,continue_batch):
        
        #Updated Q values:
        q_targets = self._get_q_targets(observation_batch, continue_batch, reward_batch)

        # Train critic model
        with tf.GradientTape() as tape:
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(q_targets - critic_value))
        
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

         # Train agent model
        with tf.GradientTape() as tape:
            actions = self.agent(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            agent_loss = -tf.math.reduce_mean(critic_value)
        
        agent_grad = tape.gradient(agent_loss, self.agent.trainable_variables)
        self.agent_optimizer.apply_gradients(zip(agent_grad, self.agent.trainable_variables))
    
        # Update targets models
        self.update_target(self.target_agent.variables, self.agent.variables)
        self.update_target(self.target_critic.variables, self.critic.variables)       

    @tf.function
    def update_target(self, target_weights, weights):
        """
        Updates the target model to slowly track the main models
        :return: None
        """
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def save_models(self, filename):
        self.agent.save(filename+"/agent")
        self.critic.save(filename+"/critic")
        self.target_agent.save(filename+"/target_agent")
        self.target_critic.save(filename+"/target_critic")

    def load_models(self, filename):
        self.agent = tf.keras.models.load_model(filename+'/agent')
        self.target_agent = tf.keras.models.load_model(filename+'/target_agent')
        self.critic = tf.keras.models.load_model(filename+'/critic')
        self.target_critic = tf.keras.models.load_model(filename+'/critic')

#For testing
if __name__ == '__main__':
    state_size  = 3
    action_size = 2
    up_bounds = np.zeros(action_size)
    low_bounds = np.ones(action_size)

    agent = DDPG(state_size=state_size, action_size=action_size,
                  lower_bounds=low_bounds, upper_bounds=up_bounds,
                  verbose=True)

    print(agent.policy(np.ones(state_size)))