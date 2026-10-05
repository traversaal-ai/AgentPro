from typing import List, Optional
import requests
import json
import openai
from .tools import Tool
from .agent import Action, Observation, ThoughtStep, AgentResponse
from .model import ModelClient, create_model

import re
from datetime import datetime



class ReactAgent:
    def __init__(self, model: Optional[ModelClient] = None, tools: List[Tool] = None, custom_system_prompt: str = None, max_iterations: int = 20):

        self.client = model or create_model(provider="openai")

        self.max_iterations = max_iterations

        # Get Tool Detail
        self.tools = tools or []
        self.tool_registry = {tool.action_type: tool for tool in self.tools}
        # Build dynamic system prompt
        tools_description = "\n\n".join(tool.get_tool_description() for tool in self.tools)
        tool_names = ", ".join(tool.action_type for tool in self.tools)

        # Get current date here
        current_date = datetime.now().strftime("%B %d, %Y")

        # Default opening sentence
        default_opening = "You are an AI assistant that follows the ReAct (Reasoning + Acting) pattern."
        # Use custom system prompt if provided, otherwise use the default
        if custom_system_prompt:
            user_system_prompt = custom_system_prompt
        else:
            user_system_prompt = default_opening

        self.system_prompt = f"""{user_system_prompt}
        
Your goal is to help users by breaking down complex tasks into a series of thought-out steps and actions.

You have access to these tools: {tool_names}

{tools_description}

Your task is to:
1. Think about what action is required — Thought.
2. Take an appropriate action — Action.
3. Repeat Thought/Action as needed until you find the final answer.

### Format (Choose only one per response)

Option 1 — When action is needed:
Thought: Your reasoning about action and observation.
Action: {{"action_type": "<action_type>", "input": <input_data>}}

Option 2 — When you're confident in the final response:
Thought: Now I know the answer that will be given in Final Answer.
Final Answer: Provide a complete, well-structured response that directly addresses the original question.

### Important:
- Think step-by-step.
- Never provide both Action and Final Answer or multiple Action in the same response.
- Use available tools wisely.
- If stuck, reflect and retry but never hallucinate.
- If observation is empty or not related, reflect and retry but never hallucinate.
- The current date is {current_date}.
- If you follow the format strictly, you will be recognized as an excellent and trustworthy AI assistant.
"""

    def _format_history(self, thought_process: List[ThoughtStep]) -> str:
        history = ""
        for step in thought_process:
            if step.pause_reflection:
                history += f"PAUSE: {step.pause_reflection}\n"
            if step.thought:
                history += f"Thought: {step.thought}\n"
            if step.action:
                history += f"Action: {step.action.model_dump_json()}\n"
            if step.observation:
                history += f"Observation: {step.observation.result}\n"
        return history

    def execute_tool(self, action: Action) -> str:
        tool = self.tool_registry.get(action.action_type)
        if not tool:
            return f"Error: Unknown action type '{action.action_type}'"
        
        try:
            return tool.run(action.input)
        except Exception as e:
            return f"Error running tool '{action.action_type}': {e}"

    def _get_llm_response(self, prompt: str) -> str:
        if not self.client:
            raise ValueError("❌ LLM client not initialized")
        
        return self.client.chat_completion(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            )

    def run(self, query: str) -> AgentResponse:
        thought_process: List[ThoughtStep] = []
        printed_prompt = False  # <<< ADD A FLAG
        iterations_count = 0
        

        while iterations_count < self.max_iterations:
            iterations_count += 1
            print("=" * 50 + f" Iteration {iterations_count} ")
            
            # Initialize placeholders at the beginning
            thought = None
            action = None
            observation = None
            pause_reflection = None

            # Get the System Prompt with History (Whole thought process)
            prompt = f"{self.system_prompt}\n\nQuestion: {query}\n\n"
            if thought_process:
                prompt += self._format_history(thought_process)
            prompt += "\nNow continue with next steps by strictly following the required format.\n"

            # Print whole System Prompt once in the start
            if not printed_prompt:
                print("✅  [Debug] Sending System Prompt (with history) to LLM:")
                print(prompt)
                print("=" * 50)
                printed_prompt = True  # <<< Set flag True after printing

            # Run LLM model
            if self.client:
                step_text = self._get_llm_response(prompt)
            else:
                return AgentResponse(
                    thought_process=thought_process,
                    final_answer="❌ No LLM is Connected. Please set and pass the OPENAI_API_KEY to AgentPro."
                )

            print("🤖 [Debug] Step LLM Response:")
            print(step_text)
            
            if "Final Answer:" in step_text and "Action:" not in step_text:
                # Try to find last Thought before Final Answer
                thought_match = re.search(r"Thought:\s*(.*?)(?:Action:|PAUSE:|Final Answer:|$)", step_text, re.DOTALL)
                pause_match = re.search(r"PAUSE:\s*(.*?)(?:Thought:|Action:|Final Answer:|$)", step_text, re.DOTALL)
                    
                # Extract Thought if found
                if thought_match:
                    thought = thought_match.group(1).strip()
                    print("✅ Parsed Thought:", thought)

                # Extract PAUSE if found
                if pause_match:
                    pause_reflection = pause_match.group(1).strip()
                    print("✅ Parsed Pause Reflection:", pause_reflection)

                thought_process.append(ThoughtStep(
                        thought=thought,
                        pause_reflection=pause_reflection
                    ))

                # Extract Final Answer
                final_answer_match = re.search(r"Final Answer:\s*(.*)", step_text, re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    print("✅ Parsed Final Answer:", final_answer)

                return AgentResponse(
                    thought_process=thought_process,
                    final_answer=final_answer
                )
            else:
                try:
                    # Try Extracting Thought Action and Pause
                    thought_match = re.search(r"Thought:\s*(.*?)(?:Action:|PAUSE:|Final Answer:|$)", step_text, re.DOTALL)
                    action_match = re.search(r"Action:\s*(\{.*?\})(?:Observation:|PAUSE:|Thought:|Final Answer:|$)", step_text, re.DOTALL)
                    pause_match = re.search(r"PAUSE:\s*(.*?)(?:Thought:|Action:|Final Answer:|$)", step_text, re.DOTALL)

                    # Extract Thought if found
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        print("✅ Parsed Thought:", thought)

                    # Extract Action if found
                    if action_match:
                        action_text = action_match.group(1).strip()
                        print("✅ Parsed Action JSON:", action_text)

                        # Load action safely
                        action_data = json.loads(action_text)
                        action = Action(
                            action_type=action_data["action_type"],
                            input=action_data["input"]
                        )

                        # Execute action
                        result = self.execute_tool(action)
                        print("✅ Parsed Action Results:", result)
                        observation = Observation(result=result)

                    # Extract PAUSE if found
                    if pause_match:
                        pause_reflection = pause_match.group(1).strip()
                        print("✅ Parsed Pause Reflection:", pause_reflection)

                    # Record the thought step
                    thought_process.append(ThoughtStep(
                        thought=thought,
                        action=action,
                        observation=observation,
                        pause_reflection=pause_reflection
                    ))
                except Exception as e:
                    print(f"❌ Error parsing LLM response: {e}")
                    print(f"❌ Raw step text: {step_text}")
                    
                    error_message = (
                        f"Error parsing LLM response: {e}\n"
                        f"Response: {step_text}\n\n"
                        "### Response format (choose only one per response)\n\n"
                        "Option 1 — When action is needed:\n"
                        "Thought: Your reasoning about action\n"
                        "Action: {\"action_type\": \"<action_type>\", \"input\": <input_data>}\n\n"
                        "Option 2 — When you're confident in the final response:\n"
                        "Thought: Now I know the answer that will be given in Final Answer.\n"
                        "Final Answer: Provide a complete, well-structured response that directly addresses the original question."
                        )
                    
                    print("✅ Parsed Action Results:", error_message)

                    # Add error as an observation
                    observation = Observation(result=error_message)
    
                    # Record the thought step with the error observation
                    thought_process.append(ThoughtStep(
                        thought=None,
                        action=None,
                        observation=observation,
                        pause_reflection=None
                        ))
                    # Continue to the next iteration instead of returning
        
        # # If exceeded max steps
        return AgentResponse(
            thought_process=thought_process,
            final_answer="❌ Stopped after reaching maximum iterations limit."
        )
