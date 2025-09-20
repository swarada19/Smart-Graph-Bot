This project is a multi-agent system developed with LangGraph and Haystack. It has two main agents:

Retriever Agent → Uses a Haystack RAG pipeline to pull relevant details from uploaded documents.

Chart Generator Agent → Leverages a Python REPL tool to create visualizations from the retrieved data.

Workflow:

User uploads a document.

User asks a question.

Retriever Agent extracts the necessary information.

Chart Generator Agent turns that information into a chart.
