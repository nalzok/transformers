from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "relaxml/Mistral-7b-E8PRVQ-4Bit"
input_text = "It is a truth universally acknowledged that "

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
output = quantized_model.generate(**input_ids, max_new_tokens=40)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

