from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "qwen_0.5b_instruct"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_dir
)

tokeniser = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_dir
)

input_text = "Hello, Could you write me a code for binary search"

model_inputs = tokeniser(input_text, return_tensors="pt")

model_inputs['max_new_tokens'] = 1024

output = model.generate(**model_inputs)

output = output[0]

output = tokeniser.decode(output, True)

print(output[len(input_text):])
