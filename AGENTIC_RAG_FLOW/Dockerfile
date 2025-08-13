# Use the base image for LangGraph API
FROM langchain/langgraph-api:3.11

# Add the current directory to the container
ADD . /langgraph/agentic_rag_flow

# Set the working directory
WORKDIR /langgraph/agentic_rag_flow

# Install Python dependencies
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -r requirements.txt

# Set environment variables for LangGraph
ENV LANGSERVE_GRAPHS='{"agent":"langgraph/agentic_rag_flow/graph/graph.py:app"}'

# Expose the default port for the LangGraph API server
EXPOSE 8000

# Command to start the LangGraph API server
CMD ["langgraph", "up"]