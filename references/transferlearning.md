# Transferlearning Primer

[Huggingface Tutorial (NLP)](https://huggingface.co/docs/transformers/training)

 - Was ist "autoregressive" an einem Decoder?
 
 
##  Funktionsweise VisionEncoderDecoderModel

 - Checkpoints für Encode und Decoder laden (`from_pretrained`)
 - "FeatureExtractor": Rescale, normalize images for the model 
 - 
 
## Tokenizer

 - Ist der Tokenizer der Encoder in den NLP Modellen?

 - "It is important to use the same tokenizer a model was trained with"
 - Dafür wird `transformers.AutoTokenizer()` benutzt (s.u.)
   - "map" über die ganzen Daten (Arrays; das wird auch von
     scikit.train_test_split akzeptiert)
	 
 
 
> AutoTokenizer is a generic tokenizer class that will be
> instantiated as one of the tokenizer classes of the library when
> created with the
> AutoTokenizer.from_pretrained(pretrained_model_name_or_path) class
>  method. 
> 
> The from_pretrained() method takes care of returning the
> correct tokenizer class instance based on the model_type property
> of the config object, or when it’s missing, falling back to using
> pattern matching on the pretrained_model_name_or_path string:

## Questions

We want to use the Huggingface Model "TrOCR". It is a vision encoder/ text decoder transformer model which generates predictions on images of handwritten text. We want to predict the age of the person who has written the text and fine-tune the model accordingly.

1. How can we train this model? The huggingface page https://huggingface.co/docs/transformers/training uses a text-2-text model and trains it to classify texts (yelp ratings). We do not quite see how to transfer the code to our model / task.  

2. How important is the fact that the TrOCR model is pretrained on an English dataset, using an English decoder ("RoBERTa")? Can we ignore this, or does it constitute an obstacle, or should we maybe even replace the English decoder with a German one?


## Code
https://huggingface.co/transformers/v4.12.5/model_doc/visionencoderdecoder.html

```
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import torch

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# load image from the IAM dataset
url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# training
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

pixel_values = processor(image, return_tensors="pt").pixel_values
text = "hello world"
labels = processor.tokenizer(text, return_tensors="pt").input_ids
outputs = model(pixel_values=pixel_values, labels=labels)
loss = outputs.loss

# inference (generation)
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
