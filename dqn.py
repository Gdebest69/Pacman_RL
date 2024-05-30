import random
import os
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
from gym import Env


class DQN:
    def __init__(
        self,
        state_size,
        action_size,
        discount_factor: float,
        folder: str,
        replay_memory_length=50000,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.y = discount_factor
        self.folder = folder
        self.q_network = self._create_model()
        self.target_network = self._create_model()
        try:
            self.q_network.load_weights(os.path.join(self.folder, "weights.keras"))
            print("Loaded weights")
        except Exception as e:
            print(f"Can't load weights: {e}")
        finally:
            self.update_target_model()
            self.replay_memory = deque(maxlen=replay_memory_length)

    def _create_model(self):
        model = Sequential()

        # Conv Layers
        model.add(
            Conv2D(
                32,
                (8, 8),
                strides=4,
                padding="same",
                input_shape=self.state_size,
            )
        )
        model.add(Activation("relu"))

        model.add(Conv2D(64, (4, 4), strides=2, padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))

        model.compile(loss="mse", optimizer=Adam())
        return model

    def _create_batch(self, batch_size=128, to_zip=True):
        batch = random.sample(self.replay_memory, batch_size)
        if to_zip:
            (states, actions, rewards, next_states, terminals) = zip(*batch)
            # print(states)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            terminals = np.array(terminals)

            return (states, actions, rewards, next_states, terminals)
        return batch

    def update_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def predict_action(self, obs, env: Env, epsilon=0):
        if random.random() > epsilon:  # exploit
            x = np.expand_dims(obs, 0)
            q_values = self.q_network(x)[0]
            return np.argmax(q_values)
        # explore
        return env.action_space.sample()

    def add_data(
        self,
        state,
        action,
        reward,
        next_state,
        terminal,
    ):
        self.replay_memory.append(
            (
                state,
                action,
                reward,
                next_state,
                int(terminal),
            )
        )

    def save_data(self):
        self.q_network.save_weights(os.path.join(self.folder, "weights.keras"))

    def create_training_data(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminals,
    ):
        target_preds = np.max(self.target_network.predict(next_states, verbose=0))
        target_values = rewards + self.y * target_preds * (1 - terminals)

        q_values = self.q_network.predict(states, verbose=0)
        q_values[np.arange(len(q_values)), actions] = target_values
        return states, q_values

    def train_on_batch(self, batch_size):
        x, y = self.create_training_data(*self._create_batch(batch_size))
        with tf.GradientTape() as tape:
            # Forward pass
            q_values = self.q_network(x, training=True)
            # Compute loss
            loss = tf.reduce_mean(tf.square(y - q_values))

        # Get gradients
        gradients = tape.gradient(loss, self.q_network.trainable_variables)

        # Apply gradients
        self.q_network.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )
        return loss
