from lightning.pytorch.callbacks import Callback

# define saving as ONNX callback
class PeriodicONNXExportCallback(Callback):
    def __init__(self, every_n_epochs=100, input_sample=None, save_dir=None, name=None):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.input_sample = input_sample
        self.save_dir = save_dir
        self.name = name

    def on_train_epoch_end(self, trainer, pl_module):
        should_save = (trainer.current_epoch + 1) % self.every_n_epochs == 0
        # to catch bugs early, save at first epoch too
        should_save |= trainer.current_epoch == 1
        if should_save:
            # Create a snapshot ONNX path with the epoch number
            snapshot_path = (
                f"{self.save_dir}/{self.name}_epoch{trainer.current_epoch+1}.onnx"
            )
            print(
                f"Saving ONNX snapshot at epoch {trainer.current_epoch+1} to {snapshot_path}"
            )

            # Save model as ONNX
            pl_module.to_onnx(
                snapshot_path,
                self.input_sample,
                export_params=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )