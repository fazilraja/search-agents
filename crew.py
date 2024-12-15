from crewai import Agent, Task, Crew, Process
from litellm import completion
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
import instructor
from openai import AzureOpenAI
import json
import os

# Initialize Azure OpenAI configuration
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = "2023-05-15"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["SERPER_API_KEY"] = ""
os.environ["GOOGLE_CSE_ID"] = ""

# Define structured output models
class Recipe(BaseModel):
    """Structured recipe information"""
    title: str
    ingredients: List[str]
    instructions: List[str]
    cooking_time: Optional[str]
    servings: Optional[int]
    source_url: Optional[str] = Field(default="generated")
    
class RecipeResponse(BaseModel):
    """Structured response from recipe agents"""
    recipe: Recipe
    status: str = Field(default="success")
    error: Optional[str] = None

# Custom Instructor-enabled LiteLLM implementation
class InstructorLLM:
    def __init__(self, model_name: str = "gpt-35-turbo"):
        self.client = instructor.patch(
            AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION")
            )
        )
        self.model_name = model_name

    def __call__(self, messages: List[Dict], **kwargs) -> Dict:
        try:
            # Extract the task type from the messages or kwargs
            task_type = kwargs.get("task_type", "recipe_finder")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_model=RecipeResponse,
                max_retries=2,
                **kwargs
            )
            
            # Convert Pydantic model to dict for CrewAI compatibility
            return {"role": "assistant", "content": response.model_dump_json()}
            
        except Exception as e:
            return {
                "role": "assistant",
                "content": RecipeResponse(
                    recipe=Recipe(
                        title="Error",
                        ingredients=[],
                        instructions=[],
                    ),
                    status="error",
                    error=str(e)
                ).model_dump_json()
            }

# Initialize tools and LLM
llm = InstructorLLM(model_name="azure/gpt-35-turbo")
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define the agents with structured output support
recipe_finder = Agent(
    role='Recipe Finder',
    goal='Find the best recipes based on user requirements',
    backstory="""You are an expert at finding recipes online. Return results in structured format.""",
    tools=[search_tool, scrape_tool],
    llm=llm,
)

recipe_analyzer = Agent(
    role='Recipe Analyzer',
    goal='Analyze recipes and extract structured information',
    backstory="""You are an expert at analyzing recipes and extracting key information.""",
    tools=[scrape_tool],
    llm=llm,
)

recipe_formatter = Agent(
    role='Recipe Formatter',
    goal='Format recipe information into a standardized JSON structure',
    backstory="""You format recipe data into clean, standardized JSON.""",
    llm=llm,
)

# Rest of the code remains similar but expects structured output
def create_recipe_tasks(recipe_query: str) -> List[Task]:
    return [
        Task(
            description=f"""
                Find a recipe for {recipe_query} from reliable sources.
                Return in structured format with ingredients and instructions.
            """,
            agent=recipe_finder
        ),
        Task(
            description="""
                Analyze and validate the structured recipe from the previous task.
            """,
            agent=recipe_analyzer,
            context="Use the structured recipe from the previous task"
        ),
        Task(
            description="""
                Ensure recipe is in correct JSON format with all required fields.
            """,
            agent=recipe_formatter
        )
    ]

def main() -> Dict:
    """Find, analyze, and format a recipe with structured output"""
    recipe_query = "spaghetti carbonara"
    crew = Crew(
        agents=[recipe_finder, recipe_analyzer, recipe_formatter],
        tasks=create_recipe_tasks(recipe_query),
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        return json.loads(result)
    except Exception as e:
        return RecipeResponse(
            recipe=Recipe(
                title=recipe_query,
                ingredients=[],
                instructions=[],
            ),
            status="failed",
            error=str(e)
        ).model_dump()

if __name__ == "__main__":
    main()
