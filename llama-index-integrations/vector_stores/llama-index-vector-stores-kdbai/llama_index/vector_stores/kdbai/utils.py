from typing import Dict
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def default_sparse_encoder(text: str) -> Dict[int, int]:
    try:
        from transformers import BertTokenizer
        from collections import Counter
    except ImportError:
        raise ImportError(
            "Could not import transformers library. "
            'Please install transformers with `pip install "transformers"`'
        )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_text = tokenizer(text, padding=True, truncation=True, max_length=512)[
        "input_ids"
    ]
    return dict(Counter(tokenized_text))


# MATCH THE METADATA COLUMN DATA TYPE TO ITS PYTYPE
def convert_metadata_col_v2(column_name, column_type, column_value):
    try:
        if column_type == "s":
            return str(column_value)
        elif column_type == "C":
            return column_value.encode("utf-8")
        elif column_type == "p":
            return pd.to_datetime(column_value)
        elif column_type == "n":
            return pd.to_timedelta(column_value)
        return column_value.astype(column_type)
    except Exception as e:
        logger.error(
            f"Failed to convert column {column_name} to qtype {column_type}: {e}"
        )


def convert_metadata_col_v1(column, value):
    try:
        if column["pytype"] == "str":
            return str(value)
        elif column["pytype"] == "bytes":
            return value.encode("utf-8")
        elif column["pytype"] == "datetime64[ns]":
            return pd.to_datetime(value)
        elif column["pytype"] == "timedelta64[ns]":
            return pd.to_timedelta(value)
        return value.astype(column["pytype"])
    except Exception as e:
        logger.error(
            f"Failed to convert column {column['name']} to type {column['pytype']}: {e}"
        )
