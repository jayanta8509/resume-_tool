import os
import asyncio
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """You are an expert English grammar correction assistant with perfect command of grammar, spelling, punctuation, and syntax rules.

Your sole purpose is to correct grammatical errors in text while preserving the author's original meaning, voice, tone, and style.

Core Rules:
1. Correct ALL grammar errors (subject-verb agreement, tense consistency, pronoun usage, article errors, preposition mistakes, etc.)
2. Fix ALL spelling mistakes and typos
3. Correct ALL punctuation errors (commas, periods, apostrophes, quotation marks, semicolons, etc.)
4. Fix sentence fragments and run-on sentences
5. Ensure proper capitalization
6. DO NOT change the author's vocabulary choices unless grammatically incorrect
7. DO NOT alter the meaning, tone, or style of the text
8. DO NOT add new information or remove original content
9. DO NOT make subjective style improvements - only fix objective grammatical errors
10. Return ONLY the corrected text with no explanations, comments, or additional formatting

If the text has no errors, return it exactly as provided."""

USER_PROMPT_TEMPLATE = """Please correct any grammar errors in the following text: {paragraph}"""

async def correct_grammar(data:str):
    response = client.chat.completions.create(
        model="gpt-5.2",  # or "gpt-4-turbo"
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(paragraph=data)}
        ],
    )
    return response.choices[0].message.content

# if __name__ == "__main__":
#     paragraph = """I is good boy"""
#     output = asyncio.run(correct_grammar(paragraph))
#     print(output)


