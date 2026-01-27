"""
DeBERTa-based trainer for hallucination detection (NLI classification).

Fine-tunes microsoft/deberta-v3-large-mnli on ambiguity datasets
for 3-way NLI classification (entailment/neutral/contradiction).
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

from src.training.base_trainer import BaseTrainer, register_trainer
from src.training.data.nli_dataset import create_dataloader
from src.training.metrics.nli_metrics import NLIMetrics
from src.training.callbacks.checkpoint_callback import CheckpointCallback
from src.training.callbacks.early_stopping import EarlyStopping
from src.training.utils.model_utils import (
    load_model_and_tokenizer,
    get_optimizer,
    get_scheduler,
    setup_mixed_precision,
    count_parameters
)

logger = logging.getLogger(__name__)


@register_trainer("hallucination")
class HallucinationTrainer(BaseTrainer):
    """Trainer for fine-tuning DeBERTa on NLI task for hallucination detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hallucination trainer.

        Args:
            config: Training configuration
        """
        super().__init__(config)

        # Extract config sections
        model_config = config.get('model', {})
        hyper_config = config.get('hyperparameters', {})
        data_config = config.get('data', {})

        # Model settings
        self.model_name = model_config.get('base_model', 'microsoft/deberta-v3-large-mnli')
        self.num_labels = model_config.get('num_labels', 3)
        self.cache_dir = model_config.get('cache_dir', './models/training')

        # Hyperparameters
        self.learning_rate = hyper_config.get('learning_rate', 2e-5)
        self.weight_decay = hyper_config.get('weight_decay', 0.01)
        self.warmup_steps = hyper_config.get('warmup_steps', 500)
        self.batch_size = hyper_config.get('batch_size', 16)
        self.gradient_accumulation_steps = hyper_config.get('gradient_accumulation_steps', 4)
        self.max_grad_norm = hyper_config.get('max_grad_norm', 1.0)
        self.mixed_precision = hyper_config.get('mixed_precision', 'fp16')
        self.optimizer_type = hyper_config.get('optimizer', 'adamw')
        self.scheduler_type = hyper_config.get('scheduler', 'linear')
        self.noise_scale = hyper_config.get('noise_scale', 1.0)

        # Data settings
        self.max_seq_length = data_config.get('max_seq_length', 256)

        # Training state
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.global_step = 0  # Track global step across epochs for batch checkpointing

        # Callbacks
        self.checkpoint_callback = None
        self.early_stopping_callback = None

        logger.info(f"Initialized HallucinationTrainer on device: {self.device}")

    def prepare_data(self, train_data_path: str, val_data_path: str) -> None:
        """
        Prepare training and validation dataloaders.

        Args:
            train_data_path: Path to training JSONL file
            val_data_path: Path to validation JSONL file
        """
        logger.info("Preparing data loaders...")

        # Create dataloaders
        self.train_loader = create_dataloader(
            data_path=train_data_path,
            tokenizer_name=self.model_name,
            batch_size=self.batch_size,
            max_length=self.max_seq_length,
            shuffle=True,
            cache_dir=self.cache_dir
        )

        self.val_loader = create_dataloader(
            data_path=val_data_path,
            tokenizer_name=self.model_name,
            batch_size=self.batch_size,
            max_length=self.max_seq_length,
            shuffle=False,
            cache_dir=self.cache_dir
        )

        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")

    def build_model(self) -> None:
        """Build and initialize the DeBERTa model."""
        logger.info("Building model...")

        # Load model and tokenizer
        lora_config = self.config.get("model", {}).get("lora")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=self.model_name,
            num_labels=self.num_labels,
            cache_dir=self.cache_dir,
            device=self.device,
            lora_config=lora_config
        )

        # Count parameters
        num_params = count_parameters(self.model)
        logger.info(f"Model parameters: {num_params:,}")

        # Setup optimizer
        self.optimizer = get_optimizer(
            model=self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
            noise_scale=self.noise_scale
        )

        # Setup mixed precision
        if self.mixed_precision == 'fp16':
            self.scaler = setup_mixed_precision(enabled=True)

        logger.info("Model built successfully")

    def train(
        self,
        num_epochs: int,
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train
            output_dir: Directory to save checkpoints
            resume_from_checkpoint: Optional checkpoint to resume from

        Returns:
            Training history with metrics
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Output directory: {output_dir}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate total training steps
        num_train_steps = len(self.train_loader) * num_epochs // self.gradient_accumulation_steps

        # Setup scheduler
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            num_training_steps=num_train_steps,
            num_warmup_steps=self.warmup_steps,
            scheduler_type=self.scheduler_type
        )

        # Setup callbacks
        checkpoint_config = self.config.get('checkpointing', {})
        self.checkpoint_callback = CheckpointCallback(
            save_dir=checkpoint_config.get('save_dir', output_dir),
            save_strategy=checkpoint_config.get('save_strategy', 'best'),
            metric_for_best=checkpoint_config.get('metric_for_best', 'val_f1_macro'),
            mode=checkpoint_config.get('mode', 'max'),
            save_total_limit=checkpoint_config.get('save_total_limit', 3),
            save_every_n_epochs=checkpoint_config.get('save_every_n_epochs', 1),
            save_every_n_batches=checkpoint_config.get('save_every_n_batches', 500)  # Default: every 500 batches
        )

        early_stopping_config = self.config.get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            self.early_stopping_callback = EarlyStopping(
                patience=early_stopping_config.get('patience', 3),
                metric_name=early_stopping_config.get('metric', 'val_f1_macro'),
                mode=early_stopping_config.get('mode', 'max'),
                min_delta=early_stopping_config.get('min_delta', 0.0001)
            )

        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint_state = self.load_checkpoint(resume_from_checkpoint)

            # Determine start epoch
            if 'epoch' in checkpoint_state:
                # For batch checkpoints, continue from the same epoch
                # For epoch checkpoints, start from next epoch
                if 'batch_idx' in checkpoint_state:
                    start_epoch = checkpoint_state['epoch']
                    logger.info(f"Resuming mid-epoch from epoch {start_epoch + 1}")
                else:
                    start_epoch = checkpoint_state['epoch'] + 1
                    logger.info(f"Resuming from epoch {start_epoch + 1}")

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*80}")

            # Train
            train_loss = self._train_epoch(epoch)
            history['train_loss'].append(train_loss)

            # Validate
            val_loss, val_metrics = self._validate_epoch(epoch)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            # Callbacks
            self.checkpoint_callback.on_epoch_end(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                metrics=val_metrics
            )

            if self.early_stopping_callback:
                self.early_stopping_callback.on_epoch_end(epoch, val_metrics)

                if self.early_stopping_callback.stop_training():
                    logger.info("Early stopping triggered!")
                    break

        logger.info("\nTraining complete!")
        return history

    def _train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Calculate total steps for this epoch (for progress bar)
        total_batches = len(self.train_loader)
        total_steps = total_batches // self.gradient_accumulation_steps
        epoch_step = 0  # Track steps within this epoch

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [Step 0/{total_steps}]",
            leave=False
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                # Increment global step and save batch checkpoint
                self.global_step += 1
                epoch_step += 1
                current_loss = loss.item() * self.gradient_accumulation_steps

                # Update progress bar description to show optimizer steps
                progress_bar.set_description(
                    f"Epoch {epoch + 1} [Step {epoch_step}/{total_steps}]"
                )

                # Call batch checkpoint callback
                if self.checkpoint_callback:
                    self.checkpoint_callback.on_batch_end(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                        model=self.model,
                        optimizer=self.optimizer,
                        loss=current_loss,
                        extra_state={
                            'scaler': self.scaler.state_dict() if self.scaler else None,
                            'scheduler': self.scheduler.state_dict() if self.scheduler else None
                        }
                    )

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'step': self.global_step
            })

        avg_loss = total_loss / num_batches
        logger.info(f"Train Loss: {avg_loss:.4f}")

        return avg_loss

    def _validate_epoch(self, epoch: int) -> tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average validation loss, metrics dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        metrics_tracker = NLIMetrics()

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {epoch + 1}",
            leave=False
        )

        with torch.no_grad():
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

                # Predictions
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)

                # Update metrics
                metrics_tracker.update(
                    predictions=predictions.cpu().tolist(),
                    labels=labels.cpu().tolist(),
                    probabilities=probs.cpu().tolist()
                )

                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        metrics = metrics_tracker.compute()

        logger.info(f"Val Loss: {avg_loss:.4f}")
        logger.info(f"Val Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Val F1 (macro): {metrics['f1_macro']:.4f}")
        if 'ece' in metrics:
            logger.info(f"Val ECE: {metrics['ece']:.4f}")
        if 'brier' in metrics:
            logger.info(f"Val Brier: {metrics['brier']:.4f}")

        return avg_loss, metrics

    def evaluate(self, data_path: str) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            data_path: Path to test JSONL file

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on: {data_path}")

        # Create test dataloader
        test_loader = create_dataloader(
            data_path=data_path,
            tokenizer_name=self.model_name,
            batch_size=self.batch_size,
            max_length=self.max_seq_length,
            shuffle=False,
            cache_dir=self.cache_dir
        )

        self.model.eval()
        metrics_tracker = NLIMetrics()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)

                metrics_tracker.update(
                    predictions=predictions.cpu().tolist(),
                    labels=labels.cpu().tolist(),
                    probabilities=probs.cpu().tolist()
                )

        metrics = metrics_tracker.compute()
        metrics_tracker.log_metrics(prefix="Test")

        return metrics

    def save_checkpoint(
        self,
        output_dir: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save model checkpoint.

        Args:
            output_dir: Directory to save checkpoint
            epoch: Current epoch
            metrics: Optional metrics to save
        """
        checkpoint_path = Path(output_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_path / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }

        state_path = checkpoint_path / "training_state.pt"
        torch.save(state, state_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)

        # Load model
        model_path = checkpoint_path / "model.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load optimizer
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists() and self.optimizer:
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))

        # Load training state
        state_path = checkpoint_path / "training_state.pt"
        state = {}
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)

            # Restore global step for batch checkpoints
            if 'global_step' in state:
                self.global_step = state['global_step']
                logger.info(f"Resuming from global step {self.global_step}")

            # Restore scheduler state
            if self.scheduler and 'scheduler' in state and state['scheduler']:
                self.scheduler.load_state_dict(state['scheduler'])

            # Restore scaler state
            if self.scaler and 'scaler' in state and state['scaler']:
                self.scaler.load_state_dict(state['scaler'])

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(f"Checkpoint epoch: {state.get('epoch', 'N/A')}, batch: {state.get('batch_idx', 'N/A')}")

        return state
