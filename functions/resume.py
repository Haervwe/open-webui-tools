"""
title: Resume_analyzer
author: Haervwe
author_url: https://github.com/Haervwe
version: 0.0.2
important note: this script requires a database for resumes , you can download the one im using on https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset?resource=download 
            and either you put it as is on /app/backend/data/UpdatedResumeDataSet.csv or change the  dataset_path in Valves.
            if websearch is setted you must provide (for now the api key for this rapidapi endpoint https://rapidapi.com/Pat92/api/jobs-api14)

TODO: make database download automatic.

Call for Help: this script is made for testing purpuses in the context of a larger project aimed to create a set of LLM assisntants 
for helping in job search and career advice based on ground truth examples and proffesional experice, all open sourced, so
every feature you might think is useful, every bug or output , model that worked the best for you , any information really you wanna share will be greatly appeciated!
you can contact me through  github
"""


import logging
import json
import asyncio
from typing import Dict, List, Callable, Awaitable
from pydantic import BaseModel, Field
from dataclasses import dataclass
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
import pandas as pd
import requests
from googleapiclient.discovery import build

name = "Resume"


def setup_logger():
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()


@dataclass
class User:
    id: str
    email: str
    name: str
    role: str


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: User
    __model__: str

    class Valves(BaseModel):
        Model: str = Field(default="", description="Model tag")
        Dataset_path: str = Field(
            default="/app/backend/data/UpdatedResumeDataSet.csv",
            description="""Path tho the dataset for CSV, the script assumes two coloums "Category" and "Resume" """,
        )
        RapidAPI_key: str = Field(default="", description="Your  jobs RapidAPI Key")
        web_search: bool = Field(
            default=False, desciption="Activates web search for relevant job postings."
        )
        Temperature: float = Field(default=1, description="Model temperature")
        system_prompt_tags: str = Field(
            default="""Analyze the provided resume and identify the most relevant categories that describe the candidate's qualifications and experience.  
            Consider both hard skills (technical proficiencies) and soft skills (communication, teamwork, leadership, etc.).  
            The returned categories should be chosen from the provided list and formatted as a comma-separated string.
            """,
            description="system prompt for tag generation",
        )
        system_impresion_prompt: str = Field(
            default="""You're an experienced recruiter reviewing a resume. Provide a concise and insightful first impression, focusing on the candidate's strengths and weaknesses.  
            Consider the clarity, conciseness, and overall presentation. Does the resume effectively highlight relevant skills and experience for a target role? 
            Does the language used seem authentic and tailored, or does it appear generic and potentially AI-generated? Avoid overly positive or negative assessments; focus on objective observations.
            never output the words "first impressions"
            """,
            description="system prompt for first impressions",
        )
        system_analysis_prompt: str = Field(
            default="""Imagine you are a recruiter evaluating candidates for a competitive role. Analyze the user's resume in comparison to similar resumes, 
            highlighting any weaknesses or areas for improvement that could hinder their chances. Focus on specific, actionable feedback the candidate can use to strengthen 
            their application materials.  Consider how the user's experience and skills stack up against the competition. 
            Avoid generic advice; offer tailored recommendations based on the context of similar resumes.
            """,
            description="system prompt for adversarial analysis",
        )
        system_persona_prompt: str = Field(
            default="""You are a highly skilled recruiter known for your insightful interview questions. You're preparing to interview a candidate based on the resume provided below. 
            Your goal is to assess not only the candidate's stated skills and experience but also their deeper understanding, problem-solving abilities, and personality fit for the role. 
            Craft 5 insightful interview questions that go beyond simply verifying information on the resume.  
            These questions should encourage the candidate to tell stories, demonstrate their thought processes, 
            and reveal their potential for growth. Consider the nuances of the resume and tailor the questions accordingly.
            """,
            description="System prompt for Personal Questions",
        )
        system_career_advisor_prompt: str = Field(
            default="""You are an expert carrer advisor, take all the convesation an resume analysis in to acount and provide the User with actionable steps to improve his job prospects""",
            description="carrer advise prompt",
        )
        Top_k: int = Field(default=50, description="Model top_k")
        Top_p: float = Field(default=0.8, description="Model top_p")

    def __init__(self):
        self.type = "manifold"
        self.conversation_history = []
        self.valves = self.Valves()

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def emit_message(self, message: str):
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_status(self, level: str, message: str, done: bool):
        await self.__current_event_emitter__(
            {
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": message,
                    "done": done,
                },
            }
        )

    async def get_streaming_completion(
        self, messages: List[Dict[str, str]], model: str
    ):
        form_data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": self.valves.Temperature,
            "top_k": self.valves.Top_k,
            "top_p": self.valves.Top_p,
        }
        response = await generate_chat_completions(
            form_data,
            user=self.__user__,
            bypass_filter=False,
        )
        async for chunk in response.body_iterator:
            for part in self.get_chunk_content(chunk):
                yield part

    def get_chunk_content(self, chunk):
        chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        if chunk_str.startswith("data: "):
            chunk_str = chunk_str[6:].strip()
            if chunk_str in ["[DONE]", ""]:
                return
            try:
                chunk_data = json.loads(chunk_str)
                if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                    delta = chunk_data["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
            except json.JSONDecodeError:
                logger.error(f'ChunkDecodeError: unable to parse "{chunk_str[:100]}"')

    async def generate_tags(self, resume_text: str, valid_tags: list):
        """Generates tags for a resume."""
        tag_prompt = (
            self.valves.system_prompt_tags
            + f"""
            Valid Tags: {', '.join(valid_tags)}
        """
        )
        tag_user_prompt = f"""
            Resume:
            ```
            {resume_text}
            ```
        """

        response = await generate_chat_completions(
            {
                "model": self.valves.Model or self.__model__,
                "messages": [
                    {"role": "system", "content": tag_prompt},
                    {"role": "user", "content": tag_user_prompt},
                ],
                "stream": False,
            },
            user=self.__user__,
        )
        tags = [
            tag.strip()
            for tag in response["choices"][0]["message"]["content"].split(",")
        ]
        return [tag for tag in tags if tag in valid_tags]  # Filter out invalid tags

    async def first_impression(self, resume_text: str):
        """Generates a first impression of the resume."""
        impression_prompt = self.valves.system_impresion_prompt
        impression_user_prompt = f"""
            Resume:
            ```
            {resume_text}
            ```
        """
        response = await generate_chat_completions(
            {
                "model": self.valves.Model or self.__model__,
                "messages": [
                    {"role": "system", "content": impression_prompt},
                    {"role": "user", "content": impression_user_prompt},
                ],
                "stream": False,
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]

    async def adversarial_analysis(self, user_resume: str, df, tags: list) -> str:
        """Performs an adversarial analysis of the resume against similar ones."""
        similar_resumes = (
            df[df["Category"].isin(tags)]["Resume"]
            .sample(min(3, len(df[df["Category"].isin(tags)])))
            .tolist()
        )
        if not similar_resumes:  # Handle the case where no similar resumes are found
            return "No similar resumes found for analysis."

        similar_resumes_text = "".join(
            [f"Resume {i+1}:\n{resume}\n\n" for i, resume in enumerate(similar_resumes)]
        )
        analysis_prompt = self.valves.system_analysis_prompt
        analysis_user_prompt = f"""
            User Resume:
            ```
            {user_resume}
            ```

            Similar Resumes:
            ```
            {similar_resumes_text}
            ```
        """
        response = await generate_chat_completions(
            {
                "model": self.valves.Model or self.__model__,
                "messages": [
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": analysis_user_prompt},
                ],
                "stream": False,
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]

    async def generate_interview_questions(
        self, resume_text: str, relevant_jobs: list | None
    ) -> str:
        """Generates potential interview questions based on the resume and relevant jobs."""

        if not relevant_jobs:
            # use old prompt
            persona_prompt = self.valves.system_persona_prompt
            persona_user_prompt = f"""
            Resume:
            ```
            {resume_text}
            ```
            """
        else:
            jobs_context = "\n\n".join(
                [f"- **{job['title']}**: {job['description']}" for job in relevant_jobs]
            )
            persona_prompt = self.valves.system_persona_prompt
            persona_user_prompt = f"""
                Resume:
                ```
                {resume_text}
                ```

                Relevant Job Descriptions:
                {jobs_context}
            """
        response = await generate_chat_completions(
            {
                "model": self.valves.Model or self.__model__,
                "messages": [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": persona_user_prompt},
                ],
                "stream": False,
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]

    async def search_relevant_jobs(
        self, resume_text: str, num_results: int, tags: list
    ) -> list:
        """
        Search for relevant job postings using RapidAPI Jobs API and provided tags.
        """
        # Validate and prepare tags
        search_tags = list(set(tags))  # Remove duplicates

        if not search_tags:
            logger.warning("No tags provided for job search.")
            return []

        # Construct query and parameters
        tag_query = ";".join(search_tags)
        location_query = "Argentina;Remote"

        url = "https://jobs-api14.p.rapidapi.com/v2/list"
        headers = {
            "x-rapidapi-key": self.valves.RapidAPI_key,
            "x-rapidapi-host": "jobs-api14.p.rapidapi.com",
        }
        querystring = {
            "query": tag_query,
            "location": location_query,
            "autoTranslateLocation": "false",
            "remoteOnly": "false",
            "employmentTypes": "fulltime;parttime;intern;contractor",
            "pageSize": num_results,
        }

        try:
            # Make the API request
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()

            # Parse the response
            data = response.json()

            # Validate and process job data
            jobs = []
            for job in data.get("jobs", []):
                job_info = {
                    "title": job.get("title", "N/A"),
                    "company": job.get("company", "N/A"),
                    "description": job.get("description", "N/A"),
                    "location": job.get("location", "N/A"),
                    "employmentType": job.get("employmentType", "N/A"),
                    "timeAgoPosted": job.get("timeAgoPosted", "N/A"),
                    "link": next(
                        (
                            provider.get("url", "N/A")
                            for provider in job.get("jobProviders", [])
                        ),
                        "N/A",
                    ),
                }
                jobs.append(job_info)

            return jobs

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching job postings: {e}")
            return []
        except ValueError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return []

    async def carrer_advisor_response(self, messages: list) -> str:
        system_prompt = self.valves.system_career_advisor_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
        full_response = ""
        async for chunk in self.get_streaming_completion(
            messages, model=self.valves.Model
        ):
            full_response += chunk
            await self.emit_message(chunk)
        return full_response

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = User(**__user__)
        self.__model__ = self.valves.Model
        if __task__ in (
            TASKS.TITLE_GENERATION,
            TASKS.TAGS_GENERATION,
        ):
            response = await generate_chat_completions(
                {
                    "model": self.__model__,
                    "messages": body.get("messages"),
                    "stream": False,
                },
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        dataset_path = self.valves.Dataset_path

        user_message = body.get("messages", [])[-1].get("content", "").strip()
        if len(body.get("messages", [])) > 1:
            await self.carrer_advisor_response(body.get("messages", []))
            return ""

        await self.emit_status("info", "Processing resume...", False)
        try:
            df = pd.read_csv(dataset_path)
            valid_tags = df["Category"].unique().tolist()
        except FileNotFoundError:
            await self.emit_status("error", f"Dataset not found: {dataset_path}", True)
            return ""

        tags = await self.generate_tags(user_message, valid_tags)
        first_impression = await self.first_impression(user_message)

        await self.emit_message(f"**First Impression:**\n{first_impression}")

        await self.emit_status("info", "Performing adversarial analysis...", False)

        analysis = await self.adversarial_analysis(user_message, df, tags)
        await self.emit_message("\n\n---\n\n")
        await self.emit_message(f"**Adversarial Analysis:**\n{analysis}")

        relevant_jobs = []
        if self.valves.web_search:

            relevant_jobs = await self.search_relevant_jobs(user_message, 5, tags)
            await self.emit_status("info", "Searching for relevant jobs...", False)
            if relevant_jobs:
                await self.emit_message("\n\n---\n\n")
                await self.emit_message(f"\n\n**Relevant Job Postings:**\n")
                for job in relevant_jobs:
                    await self.emit_message(
                        f"- **{job['title']}** ({job['company']})\n{job['description']}\n{job['location']}\n{job['link']}\n"
                    )

        await self.emit_status("info", "Generating interview questions...", False)
        interview_questions = await self.generate_interview_questions(
            user_message, relevant_jobs
        )
        await self.emit_message("\n\n---\n\n")
        await self.emit_message(
            f"**Potential Interview Questions**:\n{interview_questions}"
        )

        await self.emit_status("success", "Resume analysis complete.", True)
        return ""
