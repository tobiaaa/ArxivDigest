import json
import logging
import re
import tqdm
import utils


with open("src/relevancy_prompt.txt") as f:
    _RELEVANCY_PROMPT = f.read()


def encode_prompt(query, prompt_papers):
    """Encode multiple prompt instructions into a single string."""
    prompt = _RELEVANCY_PROMPT + "\n"
    prompt += query['interest']

    for idx, task_dict in enumerate(prompt_papers):
        (title, authors, abstract) = task_dict["title"], task_dict["authors"], task_dict["abstract"]
        if not title:
            raise ValueError(f"Paper at index {idx} has an empty title: {task_dict}")
        prompt += f"###\n"
        prompt += f"{idx + 1}. Title: {title}\n"
        prompt += f"{idx + 1}. Authors: {authors}\n"
        prompt += f"{idx + 1}. Abstract: {abstract}\n"
    prompt += f"\n Generate response:\n1."
    # print(prompt)
    return prompt


def post_process_chat_gpt_response(paper_data, response, threshold_score=8):
    if response is None:
        return []
    content = response['message']['content']

    # Match numbered entries like "1. {...}" using the number as the paper index.
    # This is resilient to parse failures: a bad entry for paper N does not shift
    # the association for papers N+1, N+2, ...
    scored = {}
    for match in re.finditer(r'(\d+)\.\s*(\{[^{}]*\})', content, re.DOTALL):
        idx = int(match.group(1)) - 1  # prompt numbers are 1-based
        if idx < 0 or idx >= len(paper_data):
            continue
        try:
            scored[idx] = json.loads(match.group(2))
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON for paper {idx + 1}: {match.group(2)!r}")

    selected_data = []
    for idx, item in sorted(scored.items()):
        temp = item.get("Relevancy score", 0)
        if isinstance(temp, str) and "/" in temp:
            score = int(temp.split("/")[0])
        else:
            score = int(temp)
        if score < threshold_score:
            continue
        paper_data[idx]["Relevancy score"] = score
        paper_data[idx]["Reasons for match"] = item.get("Reasons for match", "")
        selected_data.append(paper_data[idx])
    return selected_data


def process_subject_fields(subjects):
    if '\n' in subjects:
        subjects = subjects.split('\n')[1]
    all_subjects = subjects.split(";")
    all_subjects = [s.split(" (")[0] for s in all_subjects]
    return all_subjects

def generate_relevance_score(
    all_papers,
    query,
    model_name="gpt-3.5-turbo-16k",
    threshold_score=8,
    num_paper_in_prompt=4,
    temperature=0.4,
    top_p=1.0,
    sorting=True
):
    ans_data = []
    for id in tqdm.tqdm(range(0, len(all_papers), num_paper_in_prompt)):
        prompt_papers = all_papers[id:id+num_paper_in_prompt]
        prompt = encode_prompt(query, prompt_papers)

        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=128*num_paper_in_prompt, # The response for each paper should be less than 128 tokens.
            top_p=top_p,
        )
        response = utils.openai_completion(
            prompts=prompt,
            model_name=model_name,
            batch_size=1,
            decoding_args=decoding_args,
            logit_bias={"100257": -100},  # prevent the <|endoftext|> from being generated
        )

        batch_data = post_process_chat_gpt_response(prompt_papers, response, threshold_score=threshold_score)
        ans_data.extend(batch_data)

    if sorting:
        ans_data = sorted(ans_data, key=lambda x: int(x["Relevancy score"]), reverse=True)

    return ans_data

