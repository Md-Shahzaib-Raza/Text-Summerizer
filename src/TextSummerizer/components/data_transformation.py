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

from TextSummerizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from TextSummerizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True, padding='max_length')

        target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True, padding='max_length')

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings.get('attention_mask'),
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
