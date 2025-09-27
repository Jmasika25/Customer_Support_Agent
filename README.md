# Customer_Support_Agent

## Project Overview
The Customer Support Agent is an assistant build to handle issues related to technicalities, billing mulfunctions and general issues. It's build using Gradio Interface and LangGraph. It categorizes queries as either Positive, Neutral or Negative then provides the approriate responses to them. If the Query is negative, the agent escalates the issue to a hunam agent.

## Technologies used

Python

LangGraph (for workflow management)

Langchain

Groq API (LLM model)

Gradio (for UI)

dotenv (for environment variable management)

## How it works

User submits a query: Categorization (Technical, Billing, General)

Sentiment Analysis: Determines Positive, Neutral, or Negative sentiment

Routing:
If Negative: Escalated to a human agent
        
If Technical: Responds with a technical solution
        
If Billing: Provides billing-related assistance
        
If General: Answers general inquiries


