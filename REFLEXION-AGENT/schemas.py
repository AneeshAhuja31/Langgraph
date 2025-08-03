from typing import List
from pydantic import BaseModel,Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


from typing import List
from pydantic import BaseModel, Field

class AnswerQuestion(BaseModel):
    """Tool for providing detailed answers with self-reflection and search recommendations."""
    
    answer: str = Field(description="~250 word detailed answer to the question")
    
    # Flattened reflection fields instead of nested model
    missing: str = Field(description="Critique of what is missing from the answer")
    superfluous: str = Field(description="Critique of what is superfluous in the answer")
    
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer",
        min_items=1,
        max_items=3
    )

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )