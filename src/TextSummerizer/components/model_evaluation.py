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


import evaluate
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from TextSummerizer.entity import ModelEvaluationConfig

class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        column_text="article",
        column_summary="highlights",
    ):

        article_batches = list(
            self.generate_batch_sized_chunks(
                dataset[column_text], batch_size
            )
        )
        target_batches = list(
            self.generate_batch_sized_chunks(
                dataset[column_summary], batch_size
            )
        )

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)
        ):

            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128,
            )

            decoded_summaries = [
                tokenizer.decode(
                    s,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for s in summaries
            ]

            decoded_summaries = [
                d.replace("", " ") for d in decoded_summaries
            ]

            metric.add_batch(
                predictions=decoded_summaries, references=target_batch
            )

        # Compute and return ROUGE scores
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(device)

        # Loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        # 1. UPDATED: Use evaluate.load instead of load_metric
        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt["test"][0:10],
            rouge_metric,
            model_pegasus,
            tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary",
        )

        # 2. UPDATED: score returns clean floats directly (e.g., {'rouge1': 0.35, ...})
        # We can extract values directly without using .mid.fmeasure
        rouge_dict = {rn: score[rn] for rn in rouge_names}

        df = pd.DataFrame(rouge_dict, index=["pegasus"])
        df.to_csv(self.config.metric_file_name, index=False)