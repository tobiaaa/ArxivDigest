import logging
import time

import openai


def openai_structured_completion(
    prompt,
    model_name,
    response_format,
    temperature=0.2,
    max_tokens=1800,
    top_p=1.0,
    sleep_time=2,
):
    """Call OpenAI API with structured output using a Pydantic response_format."""
    client = openai.OpenAI()
    backoff = 3
    while True:
        try:
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return completion.choices[0].message.parsed
        except openai.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if not backoff:
                logging.error("Hit too many failures, exiting")
                raise
            backoff -= 1
            logging.warning("Retrying...")
            time.sleep(sleep_time)
