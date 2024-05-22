import numpy as np
import tensorflow as tf
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.networks import InvertibleNetwork, SetTransformer
from bayesflow.amortizers import AmortizedPosterior

# local imports
from src.custom_trainer import CustomTrainer
from src.tensorboard_callback import TensorBoardCallback


def run_minimal_example(log_dir):

    def simulator(parameters):
        x = parameters[:, 0]
        y = parameters[:, 1]

        z = tf.square(x) + tf.square(y)

        return tf.reshape(z, [-1, 1, 1])

    def prior(batch_size=1):
        return tf.concat([
            tf.random.uniform([batch_size, 1], 0.0, 1.0),
            tf.random.uniform([batch_size, 1], 2.0, 2.5)
        ], axis=1)

    prior = Prior(batch_prior_fun=prior)
    simulator = Simulator(batch_simulator_fun=simulator)
    generative_model = GenerativeModel(prior, simulator, simulator_is_batched=True, prior_is_batched=True)

    summary_net = SetTransformer(input_dim=1)
    inference_net = InvertibleNetwork(num_params=2)
    amortized_posterior = AmortizedPosterior(inference_net, summary_net)

    trainer = CustomTrainer(
        amortizer=amortized_posterior,
        generative_model=generative_model,
        callbacks=[TensorBoardCallback(log_dir)]  # Adding the TensorBoard callback
    )

    _ = trainer.train_online(epochs=10, iterations_per_epoch=1000, batch_size=32)
