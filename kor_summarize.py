import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration


#  Load Model and Tokenize
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
#model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
def kor_summarize(input_text, max_length=120, num_beams=5):
    if np.isnan(input_text):
        return ''
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # Generate Summary Text Ids
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(input_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        no_repeat_ngram_size=2,
        num_beams=num_beams
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return summary


df_closed_company = pd.read_csv('data/closed_company_summary.csv', index_col=0)
descs = df_closed_company.MN_BIZ_CONT

sumarized_descs = []
for desc in descs:
    response = kor_summarize(
        input_text=desc,
        max_length=7,
        num_beams=7
    )
    sumarized_descs.append(response)

print(sumarized_descs)