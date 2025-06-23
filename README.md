# Multi-Agent Mental Health Chatbot 

This project is a multi-agent mental health chatbot designed to provide empathetic and personalized support. Built with Python, Google's Gemini models, LangGraph, and Chainlit, it creates an interactive and responsive user experience.

The chatbot dynamically routes user conversations to the most appropriate agent: a 'Base Agent' for casual chats and an 'Advanced Agent' for more sensitive mental health discussions. A key feature is the 'Profile Building Agent,' which anonymously learns from each interaction to build a user profile, allowing the chatbot to remember context and offer more tailored support over time. This system ensures that users receive understanding and relevant guidance while maintaining privacy.

## Quick Start

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Set up Environment Variables

### 3. Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key to your `.env` file

### 4. Run the Application
```
chainlit run app.py -w
```

The application will be available at `http://localhost:8000`

## Project Structure

```
mental-health-chatbot/
├── app.py                 # Main Chainlit application
├── requirements.txt       # Python dependencies
├── .env.template         # Environment variables template
├── .env                  # Your environment variables (create from template)
├── user_profiles/        # Directory for user profiles (auto-created)
│   └── profiles.json     # JSON file storing user profiles
└── README.md            # This file
```

## Architecture Overview

The chatbot uses a multi-agent system with LangGraph:

1. **Router Agent**: Analyzes incoming messages and routes to appropriate agent
2. **Base Agent**: Handles casual conversations (uses Gemini 1.5 Flash)
3. **Advanced Agent**: Handles sensitive mental health topics (uses Gemini 2.0 Flash Lite)
4. **Profile Building Agent**: Updates user profiles based on conversations

## Data Storage

- User profiles are stored in `user_profiles/profiles.json`
- Each user gets a unique ID for the session
- Profiles include:
  - Mood history
  - Recurring topics
  - Effective techniques
  - Session count
  - General summary


## Customization

### Modify Agent Behavior
Edit the prompt templates in `app.py` for each agent:
- `router_agent()`: Modify routing logic
- `base_agent()`: Change casual conversation style
- `advanced_agent()`: Adjust mental health support approach
- `profile_building_agent()`: Customize profile extraction

### Change Models
Update the model initialization in `app.py`:
```python
base_model = genai.GenerativeModel('gemini-1.5-flash')
advanced_model = genai.GenerativeModel('gemini-2.0-flash-lite')
```

## License
MIT License

This project is for educational and research purposes. Ensure compliance with:
- Gemini API terms of service
- Healthcare data regulations
- Mental health support guidelines

## Disclaimer

This chatbot is not a replacement for professional mental health care. Users experiencing crisis situations should contact emergency services or qualified mental health professionals immediately.
