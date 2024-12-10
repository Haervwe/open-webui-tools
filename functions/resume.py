"""
title: Resume_analyzer
author: Haervwe
author_url: https://github.com/Haervwe
version: 0.0.1
important note: this script requires a database for resumes , you can download the one im using on https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset?resource=download 
            and either you put it as is on /app/backend/data/UpdatedResumeDataSet.csv or change the  dataset_path in Valves.
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
        GOOGLE_CSE_API_KEY: str = Field(
            default="", description="YOUR GOOGLE CSE API KEY"
        )
        GOOGLE_CSE_ID: str = Field(default="", description="YOUR GOOGLE CSE ID")
        Temperature: float = Field(default=1, description="Model temperature")
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

    async def generate_tags(self, resume_text, valid_tags):
        """Generates tags for a resume."""
        tag_prompt = f"""
            Analyze the provided resume and identify the most relevant categories that describe the candidate's qualifications and experience.  Consider both hard skills (technical proficiencies) and soft skills (communication, teamwork, leadership, etc.).  The returned categories should be chosen from the provided list and formatted as a comma-separated string.

            Valid Tags: {', '.join(valid_tags)}
        """
        tag_user_prompt=f"""
            Resume:
            ```
            {resume_text}
            ```
        """
        
        response = await generate_chat_completions(
            {
                "model": self.valves.Model or self.__model__,
                "messages": [{"role": "system", "content": tag_prompt},{"role": "user", "content": tag_user_prompt}],
                "stream": False,
            },
            user=self.__user__,
        )
        tags = [
            tag.strip()
            for tag in response["choices"][0]["message"]["content"].split(",")
        ]
        return [tag for tag in tags if tag in valid_tags]  # Filter out invalid tags

    async def first_impression(self, resume_text):
        """Generates a first impression of the resume."""
        impression_prompt = f"""
            You're an experienced recruiter reviewing a resume. Provide a concise and insightful first impression, focusing on the candidate's strengths and weaknesses.  Consider the clarity, conciseness, and overall presentation. Does the resume effectively highlight relevant skills and experience for a target role? Does the language used seem authentic and tailored, or does it appear generic and potentially AI-generated? Avoid overly positive or negative assessments; focus on objective observations.
            """
        impression_user_prompt=f"""
            Resume:
            ```
            {resume_text}
            ```
            never output the words "first impressions"
        """
        response = await generate_chat_completions(
            {
                "model": self.valves.Model or self.__model__,
                "messages": [{"role": "system", "content": impression_prompt},{"role": "user", "content": impression_user_prompt}],
                "stream": False,
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]

    async def adversarial_analysis(
        self, user_resume, df, tags
    ):  # Add df and tags as arguments
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
        analysis_prompt = f"""
            Imagine you are a recruiter evaluating candidates for a competitive role. Analyze the user's resume in comparison to similar resumes, 
            highlighting any weaknesses or areas for improvement that could hinder their chances. Focus on specific, actionable feedback the candidate can use to strengthen 
            their application materials.  Consider how the user's experience and skills stack up against the competition. 
            Avoid generic advice; offer tailored recommendations based on the context of similar resumes.
            """
        analysis_user_prompt=f"""
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
                "messages": [{"role": "system", "content": analysis_prompt},{"role": "user", "content": analysis_user_prompt}],
                "stream": False,
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]

    async def generate_interview_questions(self, resume_text, relevant_jobs):
        """Generates potential interview questions based on the resume and relevant jobs."""

        if not relevant_jobs:
            # use old prompt
            persona_prompt = f"""
            You are a highly skilled recruiter known for your insightful interview questions. You're preparing to interview a candidate based on the resume provided below. 
            Your goal is to assess not only the candidate's stated skills and experience but also their deeper understanding, problem-solving abilities, and personality fit for the role. 
            Craft 5 insightful interview questions that go beyond simply verifying information on the resume.  
            These questions should encourage the candidate to tell stories, demonstrate their thought processes, 
            and reveal their potential for growth. Consider the nuances of the resume and tailor the questions accordingly.
            """
            persona_user_prompt=f"""
            Resume:
            ```
            {resume_text}
            ```
            """
        else:
            jobs_context = "\n\n".join(
                [f"- **{job['title']}**: {job['snippet']}" for job in relevant_jobs]
            )
            persona_prompt = f"""
                You are an expert interviewer preparing questions for a candidate based on their resume and a set of relevant job descriptions.  
                Your goal is to craft insightful questions that assess the candidate's suitability for these specific roles.
                Focus on evaluating their skills, experience, and problem-solving abilities in the context of the target positions.
                Formulate 5 diverse interview questions that delve deeper than surface-level qualifications.  
                Encourage the candidate to demonstrate their critical thinking, relevant expertise, and potential to excel in these roles.
                """
            persona_user_prompt=f"""
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
                "messages": [{"role": "system", "content": persona_prompt},{"role": "user", "content": persona_user_prompt}],
                "stream": False,
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]

    async def search_relevant_jobs(self, resume_text, num_results=5):

        query_system_prompt = f"""
        Generate a concise Google search query (no explanations) to find entry-level or internship job postings relevant to the provided resume. 
        Prioritize key skills and job titles over specific technologies.  If a clear location preference is stated in the resume, include it in the query do not use " " in more than 1 query item.
        dont be overly specific, add the keyword REMOTE if the job can be done remotely , don use extensive quoting, use it only for single words such as "AI"

        Example Query:
        
        'site:linkedin.com/in/ "Ualá" AND '
        '"Argentina" AND '
        '("Recruiter" OR "HeadHunter" OR "Talent Acquisition" OR "HR" OR "Human Resources" OR '
        '"People" OR "People Partner" OR "Tech Recruiter" OR "Technical Recruiter" OR "HR Business Partner")'
        """
        
        query_user_prompt =f"""
        Resume:
        ```
        {resume_text}
        ```
        """
        response = await generate_chat_completions(
            {
                "model": self.valves.Model or self.__model__,
                "messages": [{"role": "system", "content": query_system_prompt},{"role": "user", "content": query_user_prompt}],
                "stream": False,
            },
            user=self.__user__,
        )
        search_query = response["choices"][0]["message"]["content"]
        logger.debug(search_query)
        service = build(
            "customsearch", "v1", developerKey=self.valves.GOOGLE_CSE_API_KEY
        )

        try:
            res = (
                service.cse()
                .list(
                    q=search_query,
                    cx=self.valves.GOOGLE_CSE_ID,  # Your Custom Search Engine ID
                    num=num_results,  # Number of results
                )
                .execute()
            )
            logger.debug("Web Searched")
            logger.debug(res)
            jobs = []
            for item in res.get("items", []):
                jobs.append(
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                    }
                )

            return jobs

        except Exception as e:
            logger.error(f"Error in Google Search: {e}")
            return []

    async def carrer_advisor_response (self, messages):
        system_prompt = f"""You are an expert carrer advisor, take all the convesation an resume analysis in to acount and provide the User with actionable steps to improve his job prospects"""
        messages = [
                        {"role": "system", "content": system_prompt},
                        *messages,
                    ]
        full_response = ""
        async for chunk in self.get_streaming_completion(messages,model= self.valves.Model):
            full_response += chunk
            await self.emit_message(
                chunk
            )
        
        
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
        if len(body.get("messages",[]))>1:
            await self.carrer_advisor_response(body.get("messages",[]))
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

        analysis = await self.adversarial_analysis(
            user_message, df, tags
        )
        await self.emit_message("\n\n---\n\n")
        await self.emit_message(f"**Adversarial Analysis:**\n{analysis}")
        await self.emit_status("info", "Searching for relevant jobs...", False)
        relevant_jobs = await self.search_relevant_jobs(user_message)

        if relevant_jobs:
            await self.emit_message("\n\n---\n\n")
            await self.emit_message(f"\n\n**Relevant Job Postings:**\n")
            for job in relevant_jobs:
                await self.emit_message(
                    f"- **{job['title']}** ({job['link']})\n{job['snippet']}\n"
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
