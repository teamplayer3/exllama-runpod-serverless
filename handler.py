import os
import logging
import runpod

from transformers import pipeline


pipe = None


def inference(event) -> [str]:
    global pipe

    logging.info(event)
    job_input = event["input"]
    if not job_input:
        raise ValueError("No input provided")

    prompt: str = job_input.pop("prompt")
    temperature: float = job_input.pop("temperature", 1.0)
    max_new_tokens = job_input.pop("max_new_tokens", 100)
    do_sample = job_input.pop("do_sample", False)
    top_p = job_input.pop("top_p", 0.95)
    repetition_penalty = job_input.pop("repetition_penalty", 1.0)
    n = job_input.pop("n", 1)

    output = []

    for i in range(n):
        output_text = pipe(prompt, temperature=temperature,
                           max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p, repetition_penalty=repetition_penalty)[0]["generated_text"]

        logging.debug(f"generated sample {i}/{n}: {output_text}")
        output.append(output_text)

    yield output


def load_model():
    revision = os.getenv("MODEL_REVISION", "main")
    model = os.environ["MODEL_REPO"]
    quantization = os.getenv("MODEL_QUANTIZATION", None)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=True)

    cache_dir = os.environ["TRANSFORMERS_CACHE"]

    if quantization == "GPTQ":
        from auto_gptq import AutoGPTQForCausalLM

        model = AutoGPTQForCausalLM.from_quantized(model,
                                                   use_safetensors=True,
                                                   trust_remote_code=True,
                                                   device="cuda:0",
                                                   quantize_config=None,
                                                   revision=revision,
                                                   cache_dir=cache_dir)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, cache_dir=cache_dir, device="cuda:0", trust_remote_code=True)

    pipe = pipeline("text-generation", model=model,
                    tokenizer=tokenizer, trust_remote_code=True)

    return pipe


def main():
    global pipe

    pipe = load_model()

    runpod.serverless.start(
        {"handler": inference, "return_aggregate_stream": True})


if __name__ == "__main__":
    main()
