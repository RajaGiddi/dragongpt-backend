import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import uuid
from typing import List, TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from sqlalchemy import create_engine
import json
import ast
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env.local')

df = pd.read_csv('data_collection/tms/winter-tms.csv')


engine = create_engine("sqlite:///winterTms.db")
#df.to_sql("winterTms", engine, index=False)
db = SQLDatabase(engine=engine)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked "
            "to extract, return null for the attribute's value.",
        ),
        MessagesPlaceholder("examples"),
        ("human", "{text}"),
    ]
)

class SubjectCoursePair(BaseModel):
    """Represents a subject and course pair."""
    subject: str = Field(..., description="The subject code of the course (e.g., 'CS').")
    course: int = Field(..., description="The course number (e.g., 101).")

class TMS(BaseModel):
    """Information about the Term Master Schedule."""
    subject_course_pairs: List[SubjectCoursePair] = Field(
        ..., description="A list of subject and course pairs"
    )
    excluded_days: List[str] = Field(
        ..., description="A list of days to exclude for scheduling"
    )
    start_time_limit: Optional[int] = Field(
        ..., description="The earliest time (in 24-hour format) a course can start"
    )

class Data(BaseModel):
    """Extracted data from Term Master Schedule."""
    people: List[TMS]

class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str
    tool_calls: List[BaseModel]


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.model_dump_json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages

examples = [
    (
        "I'm a freshman and would like to take CS 171, CI 102, CS 164, and ENGL 103 next term. I don't like classes on Fridays and prefer to take my classes later in the day",
        TMS(
            subject_course_pairs=[
                SubjectCoursePair(subject="CS", course=171),
                SubjectCoursePair(subject="CI", course=102),
                SubjectCoursePair(subject="CS", course=164),
                SubjectCoursePair(subject="ENGL", course=103),
            ],
            excluded_days=["F"],
            start_time_limit=12
        ),
    ),
    (
        "I'm a sophomore interested in taking MATH 220, PHYS 151, CS 101, and HIST 110. I prefer classes early in the morning and want to avoid classes on Mondays.",
        TMS(
            subject_course_pairs=[
                SubjectCoursePair(subject="MATH", course=220),
                SubjectCoursePair(subject="PHYS", course=151),
                SubjectCoursePair(subject="CS", course=101),
                SubjectCoursePair(subject="HIST", course=110),
            ],
            excluded_days=["M"],
            start_time_limit=8  # Early morning classes ;(
        ),
    ),
    (
        "I plan to take BIO 202, CHEM 105, ENGL 201, and SOC 120 next term. I don't like evening classes and prefer to have a free Wednesday.",
        TMS(
            subject_course_pairs=[
                SubjectCoursePair(subject="BIO", course=202),
                SubjectCoursePair(subject="CHEM", course=105),
                SubjectCoursePair(subject="ENGL", course=201),
                SubjectCoursePair(subject="SOC", course=120),
            ],
            excluded_days=["W"],
            start_time_limit=8  # This avoids evening classes
        ),
    ),
    (
        "I'm thinking of enrolling in PSYCH 150, ECON 101, ART 120, and STAT 250. I’d like classes to be back-to-back and avoid having classes on Thursdays.",
        TMS(
            subject_course_pairs=[
                SubjectCoursePair(subject="PSYCH", course=150),
                SubjectCoursePair(subject="ECON", course=101),
                SubjectCoursePair(subject="ART", course=120),
                SubjectCoursePair(subject="STAT", course=250),
            ],
            excluded_days=["Th"],
            start_time_limit=10 
        ),
    ),
]

messages = []

for text, tool_call in examples:
    messages.extend(
        tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
)

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0,
)


runnable = prompt | llm.with_structured_output(
    schema=Data,
    method='function_calling',
    include_raw=False,
)

response = runnable.invoke(
    {
        "text": "I'm a freshman and would like to take ANIM 140 and COM 220 next term.",
        "examples": messages,
    }
)

subject_course_pairs = []
for tms in response.people:
    subject_course_tuples = [(pair.subject, pair.course) for pair in tms.subject_course_pairs]
    subject_course_pairs.extend(subject_course_tuples)

excluded_days = []
for tms in response.people:
    excluded_days.extend(tms.excluded_days)

start_time_limit = 12

course_filters = " OR ".join(
    [f"(SubjectCode = '{subject}' AND \"CourseNo\\.\" = {course})" for subject, course in subject_course_pairs]
)

day_exclusion_filter = ""
if excluded_days:
    day_exclusion_filter = " AND ".join([f"Days_Time NOT LIKE '%{day}%'" for day in excluded_days])

time_filter = ""
if start_time_limit is not None:
    time_filter = f"CAST(SUBSTR(\"Days_Time1\", 1, INSTR(\"Days_Time1\", ':') - 1) AS INTEGER) >= {start_time_limit}"

query_parts = [f"({course_filters})"]
if day_exclusion_filter:
    query_parts.append(day_exclusion_filter)
if time_filter:
    query_parts.append(time_filter)

query = f"""
    SELECT *
    FROM winterTms
    WHERE 
        {' AND '.join(query_parts)}
"""

output_response = db.run(query)

import json
import ast

try:
    output_response = ast.literal_eval(output_response)
except Exception as e:  
    print(f"Error parsing output_response: {e}")
    output_response = []

processed_data = []
for item in output_response:
    if isinstance(item, (list, tuple)) and len(item) == 13:
        processed_data.append({
            "subject": item[0],
            "course_number": item[1],
            "instruction_type": item[2],
            "delivery_method": item[3],
            "section": item[4],
            "course_url": item[5],
            "crn": item[6],
            "course_title": item[7],
            "day": item[8],
            "time": item[9],
            "start_date": item[10],
            "final_exam": item[11],
            "instructor": item[12]
        })
    else:
        print(f"Skipping invalid entry: {item}")

with open("data_collection/generated-courses.json", "w") as json_file:
    json.dump(processed_data, json_file, indent=4)

print(json.dumps(processed_data, indent=4))
