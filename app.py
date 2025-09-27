# Building the frontend with gradio

# Importing libraries
from typing import TypedDict, Dict 
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_groq import ChatGroq
from IPython.display import display, Image 
import gradio as gr
from dotenv import load_dotenv
import os

#loading the api keys
load_dotenv()
groq_api_key = os.getenv("groq_api_key")

# Check if the key is loaded successfully
if groq_api_key:
    print("Groq API Key loaded successfully!")
else:
    print("Error: Groq API Key not found. Check your .env file.")

#Creating the state class
class State(TypedDict):
    query: str
    category: str
    sentiment: str 
    response: str 

#Defining the llm
LLM = ChatGroq(
    temperature = 0,
    groq_api_key = groq_api_key,
    model_name = "llama-3.3-70b-versatile"
)

#testing the model
result = LLM.invoke("what is deep learning?")
print(result.content)

#categorizing function
def categorize(state : State) -> State:
     "Technical, Billing, General"
     prompt = ChatPromptTemplate.from_template(
      "Categorize the following customer query into one of these categories: "
      "Technical, Billing, General. Query: {query}"
     )
     chain = prompt | LLM
     category = chain.invoke({"query" : state['query']}).content
     return {"category": category}

# analyzing sentiment function
def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | LLM
    sentiment = chain.invoke({"query": state['query']}).content  # Capture sentiment
    return {"sentiment": sentiment}  # Return sentiment instead of category

#handle technical function
def handle_technical(state : State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query : {query}"

    )
    chain = prompt | LLM
    response = chain.invoke({"query" : state['query']}).content
    return {"response" : response}

#handling billing function
def handle_billing(state : State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query : {query}"

    )
    chain = prompt | LLM
    response = chain.invoke({"query" : state['query']}).content
    return {"response" : response}

#handle general
def handle_general(state : State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query : {query}"

    )
    chain = prompt | LLM
    response = chain.invoke({"query" : state['query']}).content
    return {"response" : response}

#escalate the situation
def escalate(state : State) -> State:
    return{"response" : "This query has been escalate to a human agent due to its negative sentiment"}


# handling the routing function
def route_query(state: State) -> State:
    if state['sentiment'] == 'Negative':
        return "escalate"
    elif state['category'] == 'Technical':
        return "handle_technical"
    elif state['category'] == 'Billing':
        return "handle_billing"
    else:
        return "handle_general"
#crafting the workflow
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

#adding edges
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges("analyze_sentiment", 
route_query,{
    "handle_technical" : "handle_technical",
    "handle_billing": "handle_billing",
    "handle_general" : "handle_general",
    "escalate": "escalate"
})
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

#aadding the entry point
workflow.set_entry_point("categorize")

#compile the workflow
app = workflow.compile()

#Function to run customer support
def run_customer_support(query: str) -> dict:
    result = app.invoke({"query": query})  
    return {
        "category": result['category'], 
        "sentiment": result['sentiment'],
        "response": result['response']
    }

# Create the Gradio interface
def gradio_interface(query: str):
    result = run_customer_support(query)
    return (
        f"**Category:** {result['category']}\n\n"
        f"**Sentiment:** {result['sentiment']}\n\n"
        f"**Response:** {result['response']}"
    )

# Build the Gradio app
gui = gr.Interface(
    fn=gradio_interface,
    theme='Yntec/HaleyCH_Theme_Orange_Green',
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs=gr.Markdown(),
    title="Customer Support Agent",
    description="Provide a query and receive a categorized response. The system analyzes sentiment and routes to the appropriate support channel.",
)

# Launch the app
if __name__ == "__main__":
    gui.launch()