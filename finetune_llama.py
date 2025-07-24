#!/usr/bin/env python3
"""
Fine-tune Llama-3.1-8B-Instruct using QLoRA for research paper information extraction.
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import Dataset
from tqdm import tqdm
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingMonitor(TrainerCallback):
    """Custom callback to monitor training progress."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.losses = []
        self.eval_losses = []
        self.learning_rates = []
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Save training metrics."""
        if logs:
            # Save training loss
            if 'loss' in logs:
                self.losses.append({
                    'step': state.global_step,
                    'epoch': state.epoch,
                    'loss': logs['loss']
                })
                
            # Save evaluation loss
            if 'eval_loss' in logs:
                self.eval_losses.append({
                    'step': state.global_step,
                    'epoch': state.epoch,
                    'eval_loss': logs['eval_loss']
                })
                logger.info(f"Step {state.global_step}: Eval Loss = {logs['eval_loss']:.4f}")
                
            # Save learning rate
            if 'learning_rate' in logs:
                self.learning_rates.append({
                    'step': state.global_step,
                    'lr': logs['learning_rate']
                })
                
            # Save metrics to file
            self.save_metrics()
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics = {
            'losses': self.losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates
        }
        
        with open(self.output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Log epoch completion."""
        current_loss = self.losses[-1]['loss'] if self.losses else 'N/A'
        current_eval_loss = self.eval_losses[-1]['eval_loss'] if self.eval_losses else 'N/A'
        
        logger.info(f"Epoch {state.epoch:.1f} completed")
        logger.info(f"  Current train loss: {current_loss}")
        logger.info(f"  Current eval loss: {current_eval_loss}")
        logger.info(f"  Steps completed: {state.global_step}")
        
        # Save checkpoint info
        checkpoint_info = {
            'epoch': state.epoch,
            'global_step': state.global_step,
            'train_loss': current_loss,
            'eval_loss': current_eval_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'latest_checkpoint_info.json', 'w') as f:
            json.dump(checkpoint_info, f, indent=2)

class LlamaFineTuner:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing fine-tuning for {model_name}")
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with QLoRA configuration."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info("Tokenizer loaded successfully")
        
        logger.info("Loading model with 4-bit quantization...")
        
        # BitsAndBytes config for QLoRA
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("Model loaded with 4-bit quantization")
        
        # Configure LoRA
        logger.info("Setting up LoRA configuration...")
        
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Alpha parameter
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        trainable_params, all_params = self.model.get_nb_trainable_parameters()
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"All parameters: {all_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / all_params:.2f}%")
        
        logger.info("LoRA configuration applied successfully")
    
    def load_data(self, data_dir="./processed_data"):
        """Load training data."""
        print(f"\n📂 Loading training data from {data_dir}")
        
        data_path = Path(data_dir)
        
        # Load conversation files
        train_file = data_path / "train_conversations.json"
        val_file = data_path / "validation_conversations.json"
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_conversations = json.load(f)
        
        with open(val_file, 'r', encoding='utf-8') as f:
            val_conversations = json.load(f)
        
        print(f"📊 Train samples: {len(train_conversations)}")
        print(f"📊 Validation samples: {len(val_conversations)}")
        
        return train_conversations, val_conversations
    
    def prepare_dataset(self, conversations, max_length=3072):
        """Prepare dataset from conversation strings."""
        print(f"\n📋 Preparing dataset (max_length={max_length})...")
        
        def tokenize_function(examples):
            # Tokenize the conversations with proper padding/truncation
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",  # Fixed: pad to max_length for consistent batching
                max_length=max_length,
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            # Set padding token labels to -100 (ignored in loss calculation)
            labels = tokenized["input_ids"].copy()
            if isinstance(labels[0], list):
                # Handle batch case
                for i, label_seq in enumerate(labels):
                    labels[i] = [
                        -100 if token_id == self.tokenizer.pad_token_id else token_id 
                        for token_id in label_seq
                    ]
            
            tokenized["labels"] = labels
            return tokenized
        
        # Create dataset from conversation strings
        dataset = Dataset.from_dict({"text": conversations})
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing conversations"
        )
        
        return tokenized_dataset
    
    def setup_trainer(self, train_dataset, val_dataset, output_dir="./results"):
        """Setup the trainer."""
        print(f"\n🏋️ Setting up trainer...")
        
        # Training arguments with better monitoring
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=5,  # Increased to 5 epochs for better convergence
            learning_rate=2e-4,
            fp16=True,
            logging_steps=25,    # More frequent logging
            eval_strategy="steps",
            eval_steps=200,      # Evaluate every 200 steps
            save_steps=400,      # Save every 400 steps (multiple of eval_steps)
            save_total_limit=5,  # Keep more checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
            run_name=f"llama-arxiv-extraction-{datetime.now().strftime('%Y%m%d-%H%M')}",
            logging_dir=f"{output_dir}/logs",  # TensorBoard logs
            logging_first_step=True,
            logging_nan_inf_filter=True,
            dataloader_num_workers=2,
            disable_tqdm=False
        )
        
        # Data collator - simpler since we're padding to max_length
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None  # Not needed since we pad to max_length
        )
        
        # Create trainer with monitoring
        monitor = TrainingMonitor(output_dir)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            callbacks=[monitor]  # Add monitoring callback
        )
        
        logger.info("Trainer configured with monitoring")
        return trainer
    
    def train(self, data_dir="./processed_data", output_dir="./results"):
        """Main training function."""
        logger.info("Starting fine-tuning process...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load data
        train_conversations, val_conversations = self.load_data(data_dir)
        
        # Prepare datasets (tokenize the conversation strings)
        train_dataset = self.prepare_dataset(train_conversations)
        val_dataset = self.prepare_dataset(val_conversations)
        
        logger.info(f"Prepared train samples: {len(train_dataset)}")
        logger.info(f"Prepared val samples: {len(val_dataset)}")
        
        # Setup trainer
        trainer = self.setup_trainer(train_dataset, val_dataset, output_dir)
        
        # Print training info
        total_steps = len(train_dataset) // (trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps) * trainer.args.num_train_epochs
        logger.info(f"Training configuration:")
        logger.info(f"  Total epochs: {trainer.args.num_train_epochs}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Batch size per device: {trainer.args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {trainer.args.learning_rate}")
        logger.info(f"  Warmup steps: {trainer.args.warmup_steps}")
        logger.info(f"  Eval steps: {trainer.args.eval_steps}")
        logger.info(f"  Save steps: {trainer.args.save_steps}")
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Print final statistics
        final_logs = trainer.state.log_history
        if final_logs:
            train_losses = [log.get('loss') for log in final_logs if 'loss' in log]
            eval_losses = [log.get('eval_loss') for log in final_logs if 'eval_loss' in log]
            
            logger.info("Training completed successfully!")
            logger.info(f"Final training loss: {train_losses[-1] if train_losses else 'N/A':.4f}")
            logger.info(f"Final validation loss: {eval_losses[-1] if eval_losses else 'N/A':.4f}")
            logger.info(f"Best validation loss: {min(eval_losses) if eval_losses else 'N/A':.4f}")
            logger.info(f"Total training time: {trainer.state.train_time:.2f} seconds")
        
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Training logs saved to: {output_dir}/logs")
        logger.info(f"Training metrics saved to: {output_dir}/training_metrics.json")
        
        return trainer

def main():
    """Main function."""
    
    # Configuration
    DATA_DIR = "./processed_data"
    OUTPUT_DIR = "./finetuned_model"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Check if data exists
    if not Path(DATA_DIR).exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        logger.info("Please run prepare_data_for_finetuning.py first!")
        return
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be very slow on CPU.")
        logger.info("Consider using Google Colab or a machine with GPU.")
    
    # Initialize wandb (optional)
    if os.getenv("WANDB_API_KEY"):
        wandb.init(
            project="llama-arxiv-extraction",
            name=f"run-{datetime.now().strftime('%Y%m%d-%H%M')}"
        )
        logger.info("Weights & Biases logging enabled")
    else:
        logger.info("Weights & Biases logging disabled (no API key found)")
    
    # Create fine-tuner and start training
    finetuner = LlamaFineTuner(MODEL_NAME)
    trainer = finetuner.train(DATA_DIR, OUTPUT_DIR)
    
    logger.info("Fine-tuning completed successfully!")
    
    # Final summary
    if trainer.state.log_history:
        train_losses = [log.get('loss') for log in trainer.state.log_history if 'loss' in log]
        eval_losses = [log.get('eval_loss') for log in trainer.state.log_history if 'eval_loss' in log]
        
        logger.info("FINAL TRAINING SUMMARY:")
        logger.info(f"  Total epochs completed: {trainer.state.epoch}")
        logger.info(f"  Total steps: {trainer.state.global_step}")
        logger.info(f"  Final train loss: {train_losses[-1] if train_losses else 'N/A':.4f}")
        logger.info(f"  Final eval loss: {eval_losses[-1] if eval_losses else 'N/A':.4f}")
        logger.info(f"  Best eval loss: {min(eval_losses) if eval_losses else 'N/A':.4f}")
        logger.info(f"  Training time: {trainer.state.train_time:.2f} seconds")

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# Fine-tune Llama-3.1-8B-Instruct using QLoRA for research paper information extraction.
# """

# import os
# import json
# import torch
# from datetime import datetime
# from pathlib import Path

# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling
# )
# from peft import (
#     LoraConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training
# )
# from datasets import Dataset
# from tqdm import tqdm
# import wandb

# class LlamaFineTuner:
#     def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
#         self.model_name = model_name
#         self.tokenizer = None
#         self.model = None
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         print(f"🚀 Initializing fine-tuning for {model_name}")
#         print(f"📱 Device: {self.device}")
#         if torch.cuda.is_available():
#             print(f"🔥 GPU: {torch.cuda.get_device_name()}")
#             print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
#     def load_model_and_tokenizer(self):
#         """Load model and tokenizer with QLoRA configuration."""
#         print("\n📥 Loading tokenizer...")
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name,
#             trust_remote_code=True,
#             padding_side="right"
#         )
        
#         # Set pad token
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
#         print("✅ Tokenizer loaded")
        
#         print("\n📥 Loading model with 4-bit quantization...")
        
#         # BitsAndBytes config for QLoRA
#         from transformers import BitsAndBytesConfig
        
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#         )
        
#         # Load model
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             quantization_config=bnb_config,
#             device_map="auto",
#             trust_remote_code=True,
#             torch_dtype=torch.float16,
#             attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
#         )
        
#         # Prepare model for training
#         self.model = prepare_model_for_kbit_training(self.model)
        
#         print("✅ Model loaded with 4-bit quantization")
        
#         # Configure LoRA
#         print("\n⚙️ Setting up LoRA configuration...")
        
#         lora_config = LoraConfig(
#             r=16,  # Rank
#             lora_alpha=32,  # Alpha parameter
#             target_modules=[
#                 "q_proj", "k_proj", "v_proj", "o_proj",
#                 "gate_proj", "up_proj", "down_proj"
#             ],
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM"
#         )
        
#         # Apply LoRA
#         self.model = get_peft_model(self.model, lora_config)
#         self.model.print_trainable_parameters()
        
#         print("✅ LoRA configuration applied")
    
#     def load_data(self, data_dir="./processed_data"):
#         """Load training data."""
#         print(f"\n📂 Loading training data from {data_dir}")
        
#         data_path = Path(data_dir)
        
#         # Load conversation files
#         train_file = data_path / "train_conversations.json"
#         val_file = data_path / "validation_conversations.json"
        
#         with open(train_file, 'r', encoding='utf-8') as f:
#             train_conversations = json.load(f)
        
#         with open(val_file, 'r', encoding='utf-8') as f:
#             val_conversations = json.load(f)
        
#         print(f"📊 Train samples: {len(train_conversations)}")
#         print(f"📊 Validation samples: {len(val_conversations)}")
        
#         return train_conversations, val_conversations
    
#     def prepare_dataset(self, conversations, max_length=3072):
#         """Prepare dataset from conversation strings."""
#         print(f"\n📋 Preparing dataset (max_length={max_length})...")
        
#         def tokenize_function(examples):
#             # Tokenize the conversations with proper padding/truncation
#             tokenized = self.tokenizer(
#                 examples["text"],
#                 truncation=True,
#                 padding="max_length",  # Fixed: pad to max_length for consistent batching
#                 max_length=max_length,
#                 return_tensors=None
#             )
            
#             # For causal LM, labels are the same as input_ids
#             # Set padding token labels to -100 (ignored in loss calculation)
#             labels = tokenized["input_ids"].copy()
#             if isinstance(labels[0], list):
#                 # Handle batch case
#                 for i, label_seq in enumerate(labels):
#                     labels[i] = [
#                         -100 if token_id == self.tokenizer.pad_token_id else token_id 
#                         for token_id in label_seq
#                     ]
            
#             tokenized["labels"] = labels
#             return tokenized
        
#         # Create dataset from conversation strings
#         dataset = Dataset.from_dict({"text": conversations})
        
#         # Apply tokenization
#         tokenized_dataset = dataset.map(
#             tokenize_function,
#             batched=True,
#             remove_columns=dataset.column_names,
#             desc="Tokenizing conversations"
#         )
        
#         return tokenized_dataset
    
#     def setup_trainer(self, train_dataset, val_dataset, output_dir="./results"):
#         """Setup the trainer."""
#         print(f"\n🏋️ Setting up trainer...")
        
#         # Training arguments
#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             per_device_train_batch_size=2,
#             per_device_eval_batch_size=2,
#             gradient_accumulation_steps=4,
#             num_train_epochs=3,
#             learning_rate=2e-4,
#             fp16=True,
#             logging_steps=50,
#             eval_strategy="steps",  # Fixed: use eval_strategy instead of evaluation_strategy
#             eval_steps=250,         # Fixed: changed to 250 so save_steps is multiple
#             save_steps=500,         # 500 is multiple of 250
#             save_total_limit=3,
#             load_best_model_at_end=True,
#             metric_for_best_model="eval_loss",
#             greater_is_better=False,
#             warmup_steps=100,
#             lr_scheduler_type="cosine",
#             dataloader_pin_memory=False,
#             remove_unused_columns=False,
#             report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
#             run_name=f"llama-arxiv-extraction-{datetime.now().strftime('%Y%m%d-%H%M')}"
#         )
        
#         # Data collator - simpler since we're padding to max_length
#         data_collator = DataCollatorForLanguageModeling(
#             tokenizer=self.tokenizer,
#             mlm=False,
#             pad_to_multiple_of=None  # Not needed since we pad to max_length
#         )
        
#         # Create trainer
#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset,
#             processing_class=self.tokenizer,  # Fixed: use processing_class instead of tokenizer
#             data_collator=data_collator
#         )
        
#         print("✅ Trainer configured")
#         return trainer
    
#     def train(self, data_dir="./processed_data", output_dir="./results"):
#         """Main training function."""
#         print("🎯 Starting fine-tuning process...")
        
#         # Load model and tokenizer
#         self.load_model_and_tokenizer()
        
#         # Load data
#         train_conversations, val_conversations = self.load_data(data_dir)
        
#         # Prepare datasets (tokenize the conversation strings)
#         train_dataset = self.prepare_dataset(train_conversations)
#         val_dataset = self.prepare_dataset(val_conversations)
        
#         print(f"📊 Prepared train samples: {len(train_dataset)}")
#         print(f"📊 Prepared val samples: {len(val_dataset)}")
        
#         # Setup trainer
#         trainer = self.setup_trainer(train_dataset, val_dataset, output_dir)
        
#         # Start training
#         print("\n🚀 Starting training...")
#         trainer.train()
        
#         # Save final model
#         print("\n💾 Saving final model...")
#         trainer.save_model()
#         self.tokenizer.save_pretrained(output_dir)
        
#         print("✅ Training completed!")
#         print(f"📁 Model saved to: {output_dir}")
        
#         return trainer

# def main():
#     """Main function."""
    
#     # Configuration
#     DATA_DIR = "./processed_data"
#     OUTPUT_DIR = "./finetuned_model"
#     MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    
#     # Check if data exists
#     if not Path(DATA_DIR).exists():
#         print(f"❌ Data directory not found: {DATA_DIR}")
#         print("Please run prepare_data_for_finetuning.py first!")
#         return
    
#     # Check CUDA
#     if not torch.cuda.is_available():
#         print("⚠️ CUDA not available. Training will be very slow on CPU.")
#         print("Consider using Google Colab or a machine with GPU.")
    
#     # Initialize wandb (optional)
#     if os.getenv("WANDB_API_KEY"):
#         wandb.init(
#             project="llama-arxiv-extraction",
#             name=f"run-{datetime.now().strftime('%Y%m%d-%H%M')}"
#         )
    
#     # Create fine-tuner and start training
#     finetuner = LlamaFineTuner(MODEL_NAME)
#     trainer = finetuner.train(DATA_DIR, OUTPUT_DIR)
    
#     print("\n🎉 Fine-tuning completed successfully!")
#     print(f"📊 Final training loss: {trainer.state.log_history[-1].get('train_loss', 'N/A')}")
#     print(f"📊 Final validation loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}")

# if __name__ == "__main__":
#     main()

