import os
import ssl
import certifi

cafile = certifi.where()
os.environ["SSL_CERT_FILE"] = cafile
os.environ["REQUESTS_CA_BUNDLE"] = cafile
os.environ["CURL_CA_BUNDLE"] = cafile

# Ensure default SSL context uses certifi bundle (workaround for Windows OpenSSL store issues)
def _create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=cafile or certifi.where())
    return ctx

try:
    ssl.create_default_context = _create_default_context
except Exception:
    pass

# Pre-load certs into a context to avoid load_default_certs() failures
try:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=cafile)
except Exception as e:
    print('Warning loading certs into SSLContext:', e)



from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch

from TextSummerizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(model_pegasus, 'gradient_checkpointing_enable'):
            model_pegasus.gradient_checkpointing_enable()
        
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        training_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=self.config.num_train_epochs, 
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size, 
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay, 
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy, 
            eval_steps=self.config.eval_steps, 
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=True,  # Mixed precision training for memory efficiency
            torch_empty_cache_steps=100,  # Clear cache every 100 steps
            optim="adafactor",  # More memory efficient optimizer
            learning_rate=2e-5
        )

        trainer = Trainer(
            model=model_pegasus, 
            args=training_args, 
            processing_class=tokenizer,  # Updated from 'tokenizer' to 'processing_class'
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["test"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))