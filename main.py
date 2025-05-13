import chainlit as cl
from agents import Agent,Runner,OpenAIChatCompletionsModel,AsyncOpenAI,set_tracing_disabled
from dotenv import load_dotenv,find_dotenv
import os

set_tracing_disabled(disabled=True)
provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model ="gemini-2.0-flash",
    openai_client=provider,
)
Agentic_ai_Agent = Agent(
    name="Agentic-ai-Agent",
    instructions="""You are a helpful and knowledgeable assistant specialized in Agentic AI.

        In addition to answering Agentic AI questions, you are also allowed to help users recall their previous messages from this chat session.

        If the user asks something not related to Agentic AI or the chat history, respond with:

        ❌ Error: I’m only here to help with Agentic AI and chat history. I cannot assist with other topics.""",
    model=model,
)  
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_history", [])

@cl.on_message
async def chat(message: cl.Message):
    # Display "thinking" while processing
    msg = cl.Message(
        content="Thinking...",
        )
    await msg.send()
    f = open("history.txt", "a")
    f.write("User: " + message.content + "\n")
    history = cl.user_session.get("chat_history")
    history.append({"role": "user", "content": message.content})  # ✅ FIXED


    # Run the agent with user's input
    response = Runner.run_sync(
        starting_agent=Agentic_ai_Agent,
        input=message.content
    
    )

    # Update the message with the agent's response
    msg.content = response.final_output
    await msg.send() 
    history.append([{"role": "Agentic-ai-Agent", "content": response.final_output}])
    cl.user_session.set("chat_history", history)