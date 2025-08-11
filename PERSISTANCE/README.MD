## Persistance in LangGraph

In langgraph, persistance is achieved through a built-in layer called checkpointers

- When you compile a graph with checkpointer, a snapshot of the graph's state is saved at every "super-step" which represents a transition between nodes.
- These snapshots are associated with a unique identifier called a thread ID
- The checkpointer, an implementation of the BaseCheckpointSaver interface, then stores this state to a persistant backend.