"""
title: Resume_analyzer
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.4.5
requirements: aiofiles
important note: 1. this script requires the full_document filter added in open web ui to work with attached files, you can find it here : https://openwebui.com/f/haervwe/full_document_filter or in the git hub repo
2.this script requires a database for resumes it automatically downloads it from my github but if u have trouble : , you can download the one im using on https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset?resource=download 
            and either you put it as is on /app/backend/data/UpdatedResumeDataSet.csv or change the  dataset_path in Valves.
            if websearch is setted you must provide (for now the api key for this rapidapi endpoint https://rapidapi.com/Pat92/api/jobs-api14)

Call for Help: this script is made for testing purpuses in the context of a larger project aimed to create a set of LLM assisntants 
for helping in job search and career advice based on ground truth examples and proffesional experice, all open sourced, so
every feature you might think is useful, every bug or output , model that worked the best for you , any information really you wanna share will be greatly appeciated!
you can contact me through  github.
"""

import logging
import json
from typing import Dict, List, Callable, Awaitable
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
import pandas as pd
import aiofiles
import aiohttp
import os
from open_webui.models.users import User ,Users

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


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: User
    __model__: str
    __request__: None

    class Valves(BaseModel):
        Model: str = Field(default="", description="Model tag")
        Dataset_path: str = Field(
            default="/app/backend/data/UpdatedResumeDataSet.csv",
            description="""Path to the dataset for CSV, the script assumes two coloums "Category" and "Resume" """,
        )
        Dataset_url: str = Field(
            default="https://raw.githubusercontent.com/Haervwe/open-webui-tools/main/Extras/UpdatedResumeDataSet.csv",
            description="""URL for the Resumes DB""",
        )
        RapidAPI_key: str = Field(default="", description="Your  jobs RapidAPI Key")
        web_search: bool = Field(
            default=False, desciption="Activates web search for relevant job postings."
        )
        Temperature: float = Field(default=1, description="Model temperature")
        system_prompt_tags: str = Field(
            default="""You are tasked with analyzing a resume to identify relevant categories that describe the candidate's qualifications and experience, Analyze the provided resume and identify the most relevant categories that describe the candidate's qualifications and experience.  
            Consider both hard skills (technical proficiencies) and soft skills (communication, teamwork, leadership, etc.).  
            The returned categories should be chosen from the provided list and formatted as a comma-separated string. 
            **Instructions:**
            - Only use tags from the provided list of Valid Tags.
            - Format your response as a comma-separated list with no additional text or symbols.
            - Ensure each tag exactly matches one of the Valid Tags in spelling and capitalization.""",
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
        request = (self.__request__,)
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

    async def generate_tags(self, resume_text: str, valid_tags: List[str]) -> List[str]:
        """Generates tags for a resume with improved validation."""

        # Create case-insensitive mapping of valid tags
        valid_tags_map = {tag.lower(): tag for tag in valid_tags}

        # Properly format the valid tags string
        valid_tags_str = "**Valid Tags:** " + ", ".join(valid_tags)

        system_prompt = f"""{self.valves.system_prompt_tags}
    
    {valid_tags_str}
    """

        tag_user_prompt = f"""
        Resume:
        ```
        {resume_text}
        ```
        """

        try:
            response = await generate_chat_completions(
                request=self.__request__,
                form_data={
                    "model": self.valves.Model or self.__model__,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": tag_user_prompt},
                    ],
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "stream": False,
                },
                user=self.__user__,
            )

            raw_content = response["choices"][0]["message"]["content"]
            logger.debug(f"Raw tag generation response: {raw_content}")

            # Split by comma and clean up
            raw_tags = [tag.strip() for tag in raw_content.split(",")]

            # More flexible validation
            filtered_tags = []
            for tag in raw_tags:
                tag_lower = tag.lower()
                if tag_lower in valid_tags_map:
                    filtered_tags.append(valid_tags_map[tag_lower])
                else:
                    # Try to match partial tags
                    for valid_lower, valid_original in valid_tags_map.items():
                        if tag_lower in valid_lower or valid_lower in tag_lower:
                            filtered_tags.append(valid_original)
                            break

            logger.debug(f"Filtered tags: {filtered_tags}")
            return list(set(filtered_tags))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            await self.emit_status("error", f"Failed to generate tags: {e}", True)
            return []

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
            request=self.__request__,
            form_data={
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
            request=self.__request__,
            form_data={
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
            request=self.__request__,
            form_data={
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
        self, num_results: int, tags: List[str]
    ) -> List[Dict[str, str]]:
        """Search for relevant job postings using RapidAPI Jobs API and provided tags.

        Args:
            num_results (int): Number of results to get from the API.
            tags (list): List of tags for the job search.

        Returns:
            list: List of job postings from the API.
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
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, params=querystring
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

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

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching job postings: {e}")
            return []
        except ValueError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return []

    async def carrer_advisor_response(self, messages: list) -> str:
        """Defines the system prompts and calls LLM to get response after the first resume process is done and the user sends a new message.

        Args:
            messages (list): last messages from the chat

        Returns:
            str: career advisor LLM response
        """
        system_prompt = self.valves.system_career_advisor_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
        full_response = ""
        async for chunk in self.get_streaming_completion(
            self.__request__, messages, model=self.valves.Model
        ):
            full_response += chunk
            await self.emit_message(chunk)
        return full_response

    async def check_and_download_file(self, file_path: str, url: str):
        """Check if the resume DB exists, if not download it from the URL.

        Args:
            file_path (str): Path to the resume DB.
            url (str): URL to fetch the DB.
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            await self.emit_status(
                "info", f"Resume DB not found at {file_path}. Downloading...", False
            )

            # Download the file
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        content = await response.read()

                # Use aiofiles for asynchronous file writing
                async with aiofiles.open(file_path, "wb") as file:
                    await file.write(content)

                logger.info(f"File downloaded and saved to {file_path}")
            except aiohttp.ClientError as e:
                logger.error(
                    f"An error occurred while downloading the file from {url}: {e}"
                )
                await self.emit_status(
                    "error", f"Failed to download dataset: {e}", True
                )
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await self.emit_status("error", f"Unexpected error: {e}", True)
        else:
            logger.info(f"File already exists at {file_path}.")

    def parse_resume_context(self, messages: List[Dict[str, str]]) -> str:
        """Parses the resume context from the messages.

        Args:
            messages (List[Dict[str, str]]): List of messages containing roles and content.

        Returns:
            str: Combined resume and user message.
        """
        # Find the system message with the resume context
        system_message = next(
            (msg for msg in messages if msg["role"] == "system"), None
        )

        resume_content = ""
        if system_message:
            start_tag = "<source_context>"
            end_tag = "</source_context>"
            context_start = system_message["content"].find(start_tag)
            context_end = system_message["content"].find(end_tag)

            if (
                context_start != -1
                and context_end != -1
                and context_end > context_start
            ):
                # Extract the resume content
                resume_content = system_message["content"][
                    context_start + len(start_tag) : context_end
                ].strip()
            else:
                logger.warning(
                    "Incomplete or missing <source_context> tags in system message."
                )

        # Find the most recent user message
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        if resume_content:
            return f"{resume_content}\n\n{user_message}"
        return user_message

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = Users.get_user_by_id(__user__["id"])
        self.__model__ = self.valves.Model
        self.__request__ = __request__
        if __task__ and __task__ != TASKS.DEFAULT:
            try:
                response = await generate_chat_completions(
                    request=self.__request__,
                    form_data={
                        "model": self.__model__,
                        "messages": body.get("messages"),
                        "stream": False,
                    },
                    user=self.__user__,
                )
                return f"{name}: {response['choices'][0]['message']['content']}"
            except Exception as e:
                logger.error(f"Error during {__task__}: {e}")
                await self.emit_status(
                    "error", f"Failed to generate {__task__}: {e}", True
                )
                return

        dataset_path = self.valves.Dataset_path
        dataset_url = self.valves.Dataset_url
        await self.check_and_download_file(dataset_path, dataset_url)
        user_message = self.parse_resume_context(body.get("messages", []))
        if (body.get("messages", []))[-1].get("role", "").strip() != "user" and len(
            body.get("messages", [])
        ) > 1:
            logger.debug(body.get("messages", []))
            try:
                await self.carrer_advisor_response(body.get("messages", []))
            except Exception as e:
                logger.error(f"Error generating career advisor response: {e}")
                await self.emit_status(
                    "error", f"Failed to generate career advice: {e}", True
                )
            return

        await self.emit_status("info", "Processing resume...", False)
        try:
            df = pd.read_csv(dataset_path)
            required_columns = {"Category", "Resume"}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                logger.error(f"Dataset is missing required columns: {missing}")
                await self.emit_status(
                    "error", f"Dataset is missing required columns: {missing}", True
                )
                return
            valid_tags = df["Category"].unique().tolist()
        except FileNotFoundError:
            logger.error(f"Dataset not found: {dataset_path}")
            await self.emit_status("error", f"Dataset not found: {dataset_path}", True)
            return ""
        except pd.errors.EmptyDataError:
            logger.error(f"Dataset is empty: {dataset_path}")
            await self.emit_status("error", f"Dataset is empty: {dataset_path}", True)
            return ""
        except Exception as e:
            logger.error(f"Unexpected error loading dataset: {e}")
            await self.emit_status(
                "error", f"Unexpected error loading dataset: {e}", True
            )
            return

        try:
            tags = await self.generate_tags(user_message, valid_tags)
            first_impression = await self.first_impression(user_message)
            await self.emit_message(f"**First Impression:**\n{first_impression}")
        except Exception as e:
            logger.error(f"Error during tag generation or first impression: {e}")
            await self.emit_status("error", f"Failed during analysis: {e}", True)
            return

        await self.emit_status("info", "Performing adversarial analysis...", False)
        try:
            analysis = await self.adversarial_analysis(user_message, df, tags)
            await self.emit_message("\n\n---\n\n")
            await self.emit_message(f"**Adversarial Analysis:**\n{analysis}")
        except Exception as e:
            logger.error(f"Error during adversarial analysis: {e}")
            await self.emit_status(
                "error", f"Failed during adversarial analysis: {e}", True
            )
            return

        relevant_jobs = []
        if self.valves.web_search:
            await self.emit_status("info", "Searching for relevant jobs...", False)
            try:
                relevant_jobs = await self.search_relevant_jobs(5, tags)
                if relevant_jobs:
                    await self.emit_message("\n\n---\n\n")
                    await self.emit_message(f"\n\n**Relevant Job Postings:**\n")
                    for job in relevant_jobs:
                        await self.emit_message(
                            f"- **{job['title']}** ({job['company']})\n{job['description']}\n{job['location']}\n{job['link']}\n"
                        )
            except Exception as e:
                logger.error(f"Error searching for relevant jobs: {e}")
                await self.emit_status("error", f"Failed to search for jobs: {e}", True)

        await self.emit_status("info", "Generating interview questions...", False)
        try:
            interview_questions = await self.generate_interview_questions(
                user_message, relevant_jobs
            )
            await self.emit_message("\n\n---\n\n")
            await self.emit_message(
                f"**Potential Interview Questions**:\n{interview_questions}"
            )
        except Exception as e:
            logger.error(f"Error generating interview questions: {e}")
            await self.emit_status(
                "error", f"Failed to generate interview questions: {e}", True
            )

        await self.emit_status("success", "Resume analysis complete.", True)
        return
