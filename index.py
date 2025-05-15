!pip install transformers torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_with_t5(text, max_length=120, min_length=30):
    """
    Summarizes text using Google's T5 model.
    """
    model_name = "t5-base"  
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_text = "summarize: " + text.strip().replace("\n", " ")
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    sample_article = """
    Renewable energy has become one of the fastest-growing sectors in the world, with countries investing 
    in solar, wind, and hydroelectric power. These technologies are reducing carbon emissions and offering 
    new economic opportunities. However, the shift away from fossil fuels also creates disruptions in 
    traditional industries. Addressing these transitions with proper policy and investment will be key to 
    a successful and inclusive energy transformation.
    """

    summary = summarize_with_t5(sample_article)
    print("📝 T5 Summary:\n", summary)
