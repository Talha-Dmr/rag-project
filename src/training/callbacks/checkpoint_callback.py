"""
Checkpoint callback for saving model checkpoints during training.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CheckpointCallback:
    """
    Callback for saving model checkpoints during training.

    Supports:
    - Saving best model based on metric
    - Saving every N epochs
    - Limiting total number of checkpoints
    """

    def __init__(
        self,
        save_dir: str,
        save_strategy: str = "epoch",  # "epoch" or "best"
        metric_for_best: str = "val_f1_macro",
        mode: str = "max",  # "max" or "min"
        save_total_limit: Optional[int] = 3,
        save_every_n_epochs: int = 1,
        save_every_n_batches: Optional[int] = None  # NEW: batch-level checkpointing
    ):
        """
        Initialize checkpoint callback.

        Args:
            save_dir: Directory to save checkpoints
            save_strategy: When to save ("epoch" or "best")
            metric_for_best: Metric name to track for best model
            mode: "max" to maximize metric, "min" to minimize
            save_total_limit: Maximum number of checkpoints to keep
            save_every_n_epochs: Save every N epochs (for "epoch" strategy)
            save_every_n_batches: Save every N batches for mid-epoch recovery (None to disable)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_strategy = save_strategy
        self.metric_for_best = metric_for_best
        self.mode = mode
        self.save_total_limit = save_total_limit
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_batches = save_every_n_batches

        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.best_model_path = None
        self.checkpoint_paths = []
        self.batch_checkpoint_paths = []  # Track batch checkpoints separately

        logger.info(f"CheckpointCallback initialized: save_dir={save_dir}, batch_checkpoint_interval={save_every_n_batches}")

    def on_batch_end(
        self,
        epoch: int,
        batch_idx: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: float,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Called at the end of each batch (for mid-epoch checkpointing).

        Args:
            epoch: Current epoch number
            batch_idx: Batch index within epoch
            global_step: Global step count (across all epochs)
            model: Model to save
            optimizer: Optimizer to save
            loss: Current batch loss
            extra_state: Optional extra state to save (e.g., scaler, scheduler)
        """
        if self.save_every_n_batches is None:
            return

        # Save every N batches
        if global_step > 0 and global_step % self.save_every_n_batches == 0:
            checkpoint_name = f"checkpoint-step-{global_step}"
            checkpoint_path = self.save_dir / checkpoint_name

            self._save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                metrics={'train_loss': loss},
                extra_state={
                    'global_step': global_step,
                    'batch_idx': batch_idx,
                    **(extra_state or {})
                }
            )

            # Track batch checkpoint
            self.batch_checkpoint_paths.append(checkpoint_path)

            # Keep only the latest batch checkpoint to save space
            if len(self.batch_checkpoint_paths) > 1:
                oldest_checkpoint = self.batch_checkpoint_paths.pop(0)
                self._remove_checkpoint(oldest_checkpoint)

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        extra_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            metrics: Validation metrics
            extra_state: Optional extra state to save
        """
        should_save = False
        checkpoint_name = None

        if self.save_strategy == "epoch":
            # Save every N epochs
            if (epoch + 1) % self.save_every_n_epochs == 0:
                should_save = True
                checkpoint_name = f"checkpoint-epoch-{epoch + 1}"

        elif self.save_strategy == "best":
            # Save if metric improved
            current_metric = metrics.get(self.metric_for_best)

            if current_metric is None:
                logger.warning(
                    f"Metric '{self.metric_for_best}' not found in metrics"
                )
                return

            is_better = (
                (self.mode == 'max' and current_metric > self.best_metric) or
                (self.mode == 'min' and current_metric < self.best_metric)
            )

            if is_better:
                self.best_metric = current_metric
                should_save = True
                checkpoint_name = "best_model"
                logger.info(
                    f"New best {self.metric_for_best}: {current_metric:.4f}"
                )

        if should_save:
            checkpoint_path = self.save_dir / checkpoint_name
            self._save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                metrics,
                extra_state
            )

            # Track checkpoint path
            if checkpoint_name != "best_model":
                self.checkpoint_paths.append(checkpoint_path)

                # Remove old checkpoints if limit exceeded
                if (self.save_total_limit is not None and
                        len(self.checkpoint_paths) > self.save_total_limit):
                    oldest_checkpoint = self.checkpoint_paths.pop(0)
                    self._remove_checkpoint(oldest_checkpoint)

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        extra_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save checkpoint to disk.

        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            metrics: Metrics to save
            extra_state: Extra state to save
        """
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save optimizer
        optimizer_path = checkpoint_path / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)

        # Save training state
        state = {
            'epoch': epoch,
            'metrics': metrics,
            'best_metric': self.best_metric
        }

        if extra_state:
            state.update(extra_state)

        state_path = checkpoint_path / "training_state.pt"
        torch.save(state, state_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _remove_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Remove old checkpoint.

        Args:
            checkpoint_path: Path to checkpoint to remove
        """
        if checkpoint_path.exists():
            import shutil
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed old checkpoint: {checkpoint_path}")

    def get_best_model_path(self) -> Optional[Path]:
        """
        Get path to best saved model.

        Returns:
            Path to best model or None
        """
        best_path = self.save_dir / "best_model"
        return best_path if best_path.exists() else None
