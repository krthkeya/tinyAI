# Sample Curl Requests

## /create_quantized_model
curl -X POST http://127.0.0.1:5000/create_quantized_model \
    -H "Content-Type: application/json" \
    -d '{
        "hugging_face_url" : "https://huggingface.co/Qwen/Qwen2.5-0.5B",
        "python_interpreter":  "python3",
        "package_manager" : "pip3",
        "fp_format" : "f16",
        "quantization_format" : "Q4_K_M"
    }'

## /heyllm 
curl http://127.0.0.1:5000/heyllm?prompt=hello%20write%20me%20a%20fibonnaci%20code&tokens=200&model_path=GGUF_Qwen2.5-0.5B/Q4_K_M.gguf
