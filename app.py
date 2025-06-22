import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, TypedDict
import asyncio
import google.generativeai as genai
from langgraph.graph import StateGraph, END
import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    print("Please set your GEMINI_API_KEY in .env file or environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize models
base_model = genai.GenerativeModel('gemini-1.5-flash')  # Base model for casual conversations
advanced_model = genai.GenerativeModel('gemini-2.0-flash-lite')  # Advanced model so went with flash lite for better performanc
profile_model = genai.GenerativeModel('gemini-2.0-flash-lite')  # Profile building model, since advanced model is more capable flash lite may be better suited 

# State definition for LangGraph
class ChatState(TypedDict):
    conversation_history: List[Dict[str, str]]
    user_profile: Dict
    current_input: str
    response: str
    is_sensitive: bool
    user_id: str

# JSON file storage config
PROFILES_DIR = "user_profiles"
PROFILES_FILE = os.path.join(PROFILES_DIR, "profiles.json")

os.makedirs(PROFILES_DIR, exist_ok=True)

# Utility functions for JSON file operations
def load_all_profiles() -> Dict:
    try:
        if os.path.exists(PROFILES_FILE):
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading profiles: {e}")
        return {}

def save_profile(profiles: Dict):
    try:
        with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving profiles: {e}")

def load_user_profile(user_id: str) -> Dict:
    all_profiles = load_all_profiles()
    
    if user_id not in all_profiles:
        # Create new profile
        profile = {
            "user_id": user_id,
            "last_updated": datetime.now().isoformat(),
            "summary": "New user - just started their mental health journey",
            "mood_history": [],
            "recurring_topics": [],
            "effective_techniques": [],
            "session_count": 0
        }
        all_profiles[user_id] = profile
        save_profile(all_profiles)
    
    return all_profiles[user_id]

def save_user_profile(profile: Dict):
    # Save specific user profile to JSON file
    try:
        profile["last_updated"] = datetime.now().isoformat()
        all_profiles = load_all_profiles()
        all_profiles[profile["user_id"]] = profile
        save_profile(all_profiles)
        print(f"Profile saved for user: {profile['user_id']}")
    except Exception as e:
        print(f"Error saving user profile: {e}")

# Router agent that determines conversation path
async def router_agent(state: ChatState) -> ChatState:
    current_input = state["current_input"].lower()
    
    # Sensitive keywords that trigger advanced agent
    sensitive_keywords = [
        "depressed", "depression", "anxiety", "anxious", "stressed", "stress",
        "sad", "sadness", "worried", "worry", "panic", "scared", "fear",
        "overwhelmed", "hopeless", "helpless", "help", "support", "crisis",
        "therapy", "counseling", "mental health", "suicide", "suicidal",
        "self-harm", "hurt myself", "end it", "give up", "worthless",
        "lonely", "alone", "isolated", "crying", "tears", "breakdown"
    ]
    
    # Check for sensitive content
    is_sensitive = any(keyword in current_input for keyword in sensitive_keywords)
    
    # Use lightweight LLM for additional context analysis if not obviously sensitive
    if not is_sensitive:
        try:
            prompt = f"""
            Analyze this message for emotional distress or mental health concerns: "{state['current_input']}"
            
            Respond with only "SENSITIVE" if the message indicates:
            - Emotional distress or mental health concerns
            - Need for support or help
            - Complex personal problems
            - Signs of depression, anxiety, or other mental health issues
            
            Respond with only "CASUAL" if the message is:
            - General conversation or greetings
            - Simple questions about the service
            - Small talk or light topics
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: base_model.generate_content(prompt)
            )
            is_sensitive = "SENSITIVE" in response.text.upper()
        except Exception as e:
            print(f"Router analysis error: {e}")
            # Default to sensitive if unsure
            is_sensitive = True
    
    state["is_sensitive"] = is_sensitive
    print(f"Router: {'Advanced' if is_sensitive else 'base'} agent selected")
    return state

# Base agent for casual conversations
async def base_agent(state: ChatState) -> ChatState:
    try:
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in state["conversation_history"][-5:]  # Last 5 messages for context
        ])
        
        profile = state["user_profile"]
        
        prompt = f"""
        You are a friendly, supportive mental health chatbot assistant. Keep your response brief, warm, and helpful.
        
        User Profile Summary: {profile.get('summary', 'No profile available')}
        Session Count: {profile.get('session_count', 0)}
        
        Recent Conversation:
        {conversation_context}
        
        User: {state['current_input']}
        
        Respond in a caring, conversational tone. Keep it concise but warm. If this seems like a deeper issue, 
        gently suggest they share more about how they're feeling.
        """
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: base_model.generate_content(prompt)
        )
        
        state["response"] = response.text
        print("Base agent generated response")
        return state
        
    except Exception as e:
        print(f"Base agent error: {e}")
        state["response"] = "I'm here to listen and support you. How can I help you today?"
        return state

# Advanced agent for sensitive/complex conversations
async def advanced_agent(state: ChatState) -> ChatState:
    try:
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in state["conversation_history"][-10:]  # Adding more context for complex convos
        ])
        
        profile = state["user_profile"]
        mood_history = ""
        if profile.get("mood_history"):
            recent_moods = profile["mood_history"][-3:]  # Last 3 mood entries
            mood_history = "\n".join([
                f"- {mood.get('mood', 'unknown')}: {mood.get('reason_summary', 'no details')}" 
                for mood in recent_moods
            ])
        
        prompt = f"""
        You are an empathetic, professional mental health support chatbot. Your role is to provide emotional support, 
        active listening, and helpful guidance. You are not a replacement for professional therapy.
        
        IMPORTANT: If someone expresses suicidal thoughts or immediate danger, encourage them to contact emergency services 
        or a crisis hotline immediately.
        
        User Profile:
        - Summary: {profile.get('summary', 'No profile available')}
        - Sessions: {profile.get('session_count', 0)}
        - Recurring topics: {', '.join(profile.get('recurring_topics', []))}
        - Effective techniques: {', '.join(profile.get('effective_techniques', []))}
        - Recent mood patterns:
        {mood_history or "No mood history available"}
        
        Conversation History:
        {conversation_context}
        
        User: {state['current_input']}
        
        Guidelines:
        - Be empathetic and validate their feelings
        - Use active listening techniques
        - Ask thoughtful follow-up questions when appropriate
        - Suggest coping strategies or techniques that have worked for them before
        - Encourage professional help when needed (therapist, counselor, doctor)
        - If they mention suicidal thoughts, provide crisis resources
        - Keep responses supportive but not overly long
        
        Respond with genuine care and understanding:
        """
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: advanced_model.generate_content(prompt)
        )
        
        state["response"] = response.text
        print("Advanced agent generated response")
        return state
        
    except Exception as e:
        print(f"Advanced agent error: {e}")
        state["response"] = """I understand you're going through something difficult. I'm here to listen and support you. 

Would you like to share more about what's on your mind? Sometimes talking through our feelings can help.

If you're in crisis or having thoughts of self-harm, please reach out to:
- iCall (Tata Institute of Social Sciences): +91 9152987821
- AASRA (24/7 Helpline): +91 9820466726
- Or contact your local emergency services: 100"""
        return state

# Agent that builds and updates user profile
async def profile_building_agent(state: ChatState) -> ChatState:
    try:
        # Get the latest conversation turn
        user_message = state["current_input"]
        bot_response = state["response"]
        
        current_profile = state["user_profile"]
        
        # Increment session count
        current_profile["session_count"] = current_profile.get("session_count", 0) + 1
        
        prompt = f"""
        Based on the following conversation, update the user's profile. Extract insights and patterns
        without storing personal identifiable information. Focus on emotional patterns, coping preferences,
        and general themes that could help provide better support.
        
        Current User Profile:
        {json.dumps(current_profile, indent=2)}
        
        Latest Conversation:
        User: {user_message}
        Bot: {bot_response}
        
        Instructions:
        1. Update the summary with new insights about their emotional state and patterns
        2. Add a new mood entry if emotional state is clearly expressed (mood + brief reason)
        3. Update recurring topics if new themes emerge (work, relationships, sleep, etc.)
        4. Note effective techniques if the user responds positively to suggestions
        5. Keep all information general and supportive - no personal details
        6. Limit mood_history to last 10 entries, recurring_topics to 8 items, effective_techniques to 6 items
        
        Respond with ONLY a valid JSON object containing the updated profile. No additional text.
        """
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: profile_model.generate_content(prompt)
        )
        
        try:
            # Clean response
            response_text = response.text.strip()
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                updated_profile = json.loads(json_str)
                
                # Ensuring that the required fields exist
                if "user_id" not in updated_profile:
                    updated_profile["user_id"] = state["user_id"]
                
                if "mood_history" in updated_profile:
                    updated_profile["mood_history"] = updated_profile["mood_history"][-10:]
                if "recurring_topics" in updated_profile:
                    updated_profile["recurring_topics"] = updated_profile["recurring_topics"][-8:]
                if "effective_techniques" in updated_profile:
                    updated_profile["effective_techniques"] = updated_profile["effective_techniques"][-6:]
                
                # Save the updated profile
                save_user_profile(updated_profile)
                state["user_profile"] = updated_profile
                print(f"Profile updated for user: {state['user_id']}")
            else:
                print("Could not extract JSON from profile update response")
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in profile update: {e}")
            print(f"Response text: {response.text[:200]}...")
        
    except Exception as e:
        print(f"Profile building agent error: {e}")
    
    return state

# LangGraph workflow setup
def create_workflow():
    # Creating the LangGraph workflow
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("base", base_agent)
    workflow.add_node("advanced", advanced_agent)
    workflow.add_node("profile_builder", profile_building_agent)
    
    # Define the flow
    workflow.set_entry_point("router")
    
    # Router decides which agent to use
    workflow.add_conditional_edges(
        "router",
        lambda state: "advanced" if state["is_sensitive"] else "base",
        {
            "base": "base",
            "advanced": "advanced"
        }
    )
    
    # Both agents go to profile builder
    workflow.add_edge("base", "profile_builder")
    workflow.add_edge("advanced", "profile_builder")
    
    # Profile builder ends the workflow
    workflow.add_edge("profile_builder", END)
    
    return workflow.compile()

# Initialize the workflow
app_workflow = create_workflow()

# Chainlit Event Handlers
@cl.on_chat_start
async def start():
    # Initialize chat session
    # Generate unique user ID for this session
    user_id = str(uuid.uuid4())
    cl.user_session.set("user_id", user_id)
    
    # Load user profile
    user_profile = load_user_profile(user_id)
    cl.user_session.set("user_profile", user_profile)
    
    # Initialize conversation history
    cl.user_session.set("conversation_history", [])
    
    # Welcome message
    welcome_msg = """Hello! I'm your mental health support companion. 

I'm here to listen, understand, and provide support through whatever you're experiencing. You can talk to me about:
- How you're feeling emotionally
- Stress, anxiety, or worry
- Daily challenges you're facing
- Coping strategies and techniques
- Or anything else on your mind

Our conversation is private and I'll remember our previous interactions to better support you. How are you feeling today?

**Available commands:**
- Type "show profile" to see your conversation summary
- Type "crisis help" for emergency resources"""
    
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Handle special commands
        user_input = message.content.strip().lower()
        
        if user_input == "show profile":
            await show_user_profile()
            return
        elif user_input in ["crisis help", "crisis", "help"]:
            await show_crisis_resources()
            return
        
        # Get session data
        user_id = cl.user_session.get("user_id")
        user_profile = cl.user_session.get("user_profile", {})
        conversation_history = cl.user_session.get("conversation_history", [])
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create initial state
        initial_state = ChatState(
            conversation_history=conversation_history,
            user_profile=user_profile,
            current_input=message.content,
            response="",
            is_sensitive=False,
            user_id=user_id
        )
        
        # Show thinking message
        thinking_msg = cl.Message(content="Thinking...")
        await thinking_msg.send()
        
        # Run the workflow
        final_state = await app_workflow.ainvoke(initial_state)
        
        # Add bot response to history
        conversation_history.append({
            "role": "assistant",
            "content": final_state["response"],
            "timestamp": datetime.now().isoformat()
        })
        
        # Update session data
        cl.user_session.set("conversation_history", conversation_history)
        cl.user_session.set("user_profile", final_state["user_profile"])
        
        # Update the thinking message with the actual response
        thinking_msg.content = final_state["response"]
        
        if final_state["is_sensitive"]:
            thinking_msg.content += "\n\n*Type 'crisis help' for emergency resources or 'show profile' to see your conversation summary*"
        
        await thinking_msg.update()
            
    except Exception as e:
        print(f"Error in main handler: {e}")
        await cl.Message(content="I'm sorry, I encountered an error. Please try again.").send()

async def show_user_profile():
    user_profile = cl.user_session.get("user_profile", {})
    
    profile_text = f"""**Your Profile Summary:**

**Sessions:** {user_profile.get('session_count', 0)}
**Summary:** {user_profile.get('summary', 'No summary available')}

**Recent Moods:**
"""
    
    mood_history = user_profile.get('mood_history', [])
    if mood_history:
        for mood in mood_history[-5:]:  # Last 5 moods
            profile_text += f"- {mood.get('mood', 'unknown')}: {mood.get('reason_summary', 'no details')}\n"
    else:
        profile_text += "No mood history yet\n"
    
    recurring_topics = user_profile.get('recurring_topics', [])
    effective_techniques = user_profile.get('effective_techniques', [])
    
    profile_text += f"""
**Topics We've Discussed:** {', '.join(recurring_topics) if recurring_topics else 'None yet'}
**Techniques That Help:** {', '.join(effective_techniques) if effective_techniques else 'None identified yet'}

*This profile helps me provide better support tailored to your needs.*
    """
    
    await cl.Message(content=profile_text).send()

async def show_crisis_resources():
    """Show crisis resources"""
    crisis_text = """**Crisis Resources - Available 24/7:**

**Immediate Emergency:** Call 112

iCall Tata Institute of Social Sciences: +91 9152987821
- Free, confidential mental health support via phone, email, and online chat

**AASRA:** +91 9820466726
- 24/7 helpline offering emotional support to individuals in distress

**Online Chat:** suicidepreventionlifeline.org
- Online chat support available

**Other Resources:**
- Fortis Mental Health Helpline: +91 8376804102 (for general psychological support)
- iCall Email Support: icall@tiss.edu
- In case of an emergency: Dial 112 (India's national emergency number)

**Remember:** You are not alone, and help is always available. 

*I'm also here to listen and support you through this conversation.*"""
    
    await cl.Message(content=crisis_text).send()

if __name__ == "__main__":
    print("Starting Mental Health Chatbot with Chainlit...")
    print(f"Profiles stored in: {PROFILES_FILE}")
    print("Set your GEMINI_API_KEY in .env file")
    print("Run with: chainlit run app.py -w")