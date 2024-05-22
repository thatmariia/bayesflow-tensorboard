from bayesflow.trainers import Trainer
from bayesflow.helper_functions import backprop_step, extract_current_lr, format_loss_string
from bayesflow.default_settings import TQDM_MININTERVAL
import tensorflow as tf
from tqdm import tqdm


class CustomTrainer(Trainer):
    """
    Custom Trainer class that extends the Trainer class from bayesflow.trainers.
    Added functionality: callbacks.

    Overrides the train_online method to add callbacks functionality.
    Added private method _get_logs to extract logs for a given epoch.

    :param callbacks: list of callbacks to be executed at the end of each epoch
    """

    def __init__(self, *args, callbacks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.callbacks = callbacks or []

    def train_online(
            self,
            epochs,
            iterations_per_epoch,
            batch_size,
            save_checkpoint=True,
            optimizer=None,
            reuse_optimizer=False,
            early_stopping=False,
            use_autograph=True,
            validation_sims=None,
            **kwargs
    ):
        """
        Copied from bayesflow.trainers.Trainer.train_online:
        https://bayesflow.org/_modules/bayesflow/trainers.html#Trainer.train_online
        + added callback functionality
        """

        assert self.generative_model is not None, "No generative model found. Only offline training is possible!"

        # Compile update function, if specified
        if use_autograph:
            _backprop_step = tf.function(backprop_step, reduce_retracing=True)
        else:
            _backprop_step = backprop_step

        # Create new optimizer and initialize loss history
        self._setup_optimizer(optimizer, epochs, iterations_per_epoch)
        self.loss_history.start_new_run()
        validation_sims = self._config_validation(validation_sims, **kwargs.pop("val_model_args", {}))

        # Create early stopper, if conditions met, otherwise None returned
        early_stopper = self._config_early_stopping(early_stopping, validation_sims, **kwargs)

        # Loop through training epochs
        for ep in range(1, epochs + 1):
            with tqdm(total=iterations_per_epoch, desc=f"Training epoch {ep}", mininterval=TQDM_MININTERVAL) as p_bar:
                for it in range(1, iterations_per_epoch + 1):
                    # Perform one training step and obtain current loss value
                    loss = self._train_step(batch_size, update_step=_backprop_step, **kwargs)

                    # Store returned loss
                    self.loss_history.add_entry(ep, loss)

                    # Compute running loss
                    avg_dict = self.loss_history.get_running_losses(ep)

                    # Extract current learning rate
                    lr = extract_current_lr(self.optimizer)

                    # Format for display on progress bar
                    disp_str = format_loss_string(ep, it, loss, avg_dict, lr=lr)

                    # Update progress bar
                    p_bar.set_postfix_str(disp_str, refresh=False)
                    p_bar.update(1)

            # Store and compute validation loss, if specified
            self._validation(ep, validation_sims, **kwargs)
            self._save_trainer(save_checkpoint)

            """ Added callbacks functionality begin """
            logs = self._get_logs(ep)
            for callback in self.callbacks:
                callback.on_epoch_end(ep, logs)
            """ Added callbacks functionality end """

            # Check early stopping, if specified
            if self._check_early_stopping(early_stopper):
                break

        # Remove optimizer reference, if not set as persistent
        if not reuse_optimizer:
            self.optimizer = None

        """ Added callbacks functionality begin """
        for callback in self.callbacks:
            callback.on_train_end()
        """ Added callbacks functionality end """

        return self.loss_history.get_plottable()

    def _get_logs(self, epoch):
        """
        Custom function: Extract logs for the given epoch from self.loss_history.
        """

        run_key = f"Run {self.loss_history._current_run}"
        epoch_key = f"Epoch {epoch}"
        epoch_data = self.loss_history.history.get(run_key, {}).get(epoch_key, [])

        # Convert epoch data to logs dictionary
        logs = {}
        if epoch_data:
            if isinstance(epoch_data[0], list):  # Multiple loss components
                for i, name in enumerate(self.loss_history.loss_names):
                    logs[name] = sum(data[i] for data in epoch_data) / len(epoch_data)
            else:  # Single loss component
                logs[self.loss_history.loss_names[0]] = sum(epoch_data) / len(epoch_data)

        return logs
