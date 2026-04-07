import tqdm
from typing import List
from pydantic import BaseModel
import utils


with open("src/relevancy_prompt.txt") as f:
    _RELEVANCY_PROMPT = f.read()


class PaperScore(BaseModel):
    relevancy_score: int
    reasons_for_match: str


class RelevancyBatch(BaseModel):
    papers: List[PaperScore]


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
    return prompt


def process_subject_fields(subjects):
    if '\n' in subjects:
        subjects = subjects.split('\n')[1]
    all_subjects = subjects.split(";")
    all_subjects = [s.split(" (")[0] for s in all_subjects]
    return all_subjects

def generate_relevance_score(
    all_papers,
    query,
    model_name="gpt-4o-mini",
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

        result = utils.openai_structured_completion(
            prompt=prompt,
            model_name=model_name,
            response_format=RelevancyBatch,
            temperature=temperature,
            max_tokens=128 * num_paper_in_prompt,
            top_p=top_p,
        )

        if result is None:
            continue

        for idx, paper_score in enumerate(result.papers):
            if idx >= len(prompt_papers):
                break
            score = paper_score.relevancy_score
            if score < threshold_score:
                continue
            prompt_papers[idx]["Relevancy score"] = score
            prompt_papers[idx]["Reasons for match"] = paper_score.reasons_for_match
            ans_data.append(prompt_papers[idx])

    if sorting:
        ans_data = sorted(ans_data, key=lambda x: int(x["Relevancy score"]), reverse=True)

    return ans_data
