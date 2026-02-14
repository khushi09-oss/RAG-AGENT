"""
Gemini API Wrapper for RAG Systems
===================================

A clean, production-ready wrapper for Google's Gemini AI API.
Designed specifically for building Retrieval-Augmented Generation (RAG) systems.

Author: GDG Workshop Team
License: MIT
"""

import google.generativeai as genai
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiWrapper:
    """
    Simple, robust wrapper for Google's Gemini AI API.

    Features:
    - Easy initialization with API key management
    - Configurable temperature and model selection
    - System persona support for consistent responses
    - Conversation history tracking
    - Built-in error handling
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        verbose: bool = True
    ):
        """
        Initialize the Gemini wrapper.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env variable
            model_name: Gemini model to use ('gemini-1.5-flash' or 'gemini-1.5-pro')
            temperature: Response randomness (0.0=deterministic, 1.0=creative)
            verbose: Whether to print initialization messages

        Raises:
            ValueError: If no API key is provided or found in environment

        """
        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "No Gemini API key provided.\n"
                "Either pass api_key parameter or set GEMINI_API_KEY environment variable.\n"
                "Get your key at: https://makersuite.google.com/app/apikey"
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Store configuration
        self.temperature = temperature
        self.verbose = verbose


        # Initialize tracking
        self.history = []
        self.persona = None

        if self.verbose:
            print(f"✅ Gemini initialized, (temp={temperature})")

    def set_persona(self, persona_description: str) -> None:
        """
        Set the AI's system persona/role.

        This defines how the AI should behave and respond. Useful for:
        - Setting response style (formal, casual, technical)
        - Defining expertise area
        - Enforcing response guidelines (e.g., "always cite sources")

        Args:
            persona_description: Description of the AI's role and behavior

        """
        self.persona = persona_description
        if self.verbose:
            preview = persona_description[:80] + "..." if len(persona_description) > 80 else persona_description
            print(f"✅ Persona set: {preview}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate a response from Gemini.

        Args:
            prompt: The input prompt/question
            temperature: Override default temperature for this request
            max_tokens: Maximum response length (default: 2048)

        Returns:
            Generated text response

        Raises:
            Exception: If API call fails (returns error message as string)
        """
        # Build full prompt with persona if set
        full_prompt = ""
        if self.persona:
            full_prompt = f"SYSTEM: {self.persona}\n\nUSER: {prompt}"
        else:
            full_prompt = prompt

        # Use provided temperature or default
        temp = temperature if temperature is not None else self.temperature

        try:
            # Configure generation
            config = genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40
            )

            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=config
            )

            # Extract text
            response_text = response.text

            # Track in history
            self.history.append({
                'prompt': prompt,
                'response': response_text,
                'temperature': temp,
                'model': self.model_name
            })

            return response_text

        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return error_msg

    def chat(self, message: str) -> str:
        """
        Send a message in a multi-turn conversation.

        Unlike generate(), this maintains conversation context across multiple
        calls, allowing the AI to reference previous messages.

        Args:
            message: User message in the conversation

        Returns:
            AI's response
        """
        # Create chat session on first use
        if not hasattr(self, 'chat_session'):
            self.chat_session = self.model.start_chat(history=[])

        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            return f"Chat error: {str(e)}"

    def clear_history(self) -> None:
        """
        Clear conversation history and reset chat session.

        Useful when starting a new conversation topic.
        """
        self.history = []
        if hasattr(self, 'chat_session'):
            delattr(self, 'chat_session')
        if self.verbose:
            print("✅ History cleared")

    def get_history(self) -> List[Dict]:
        """
        Get the conversation history.

        Returns:
            List of dictionaries containing prompt, response, temperature, and model
        """
        return self.history

    def get_stats(self) -> Dict:
        """
        Get wrapper statistics.

        Returns:
            Dictionary with model info and usage stats
        """
        return {
            'model': self.model_name,
            'temperature': self.temperature,
            'total_interactions': len(self.history),
            'has_persona': self.persona is not None
        }


# ============================================================================
# Demo & Testing
# ============================================================================

def demo():
    """Run a simple demo of the Gemini wrapper."""
    print("\n" + "="*70)
    print("GEMINI WRAPPER DEMO")
    print("="*70 + "\n")

    try:
        # Initialize
        llm = GeminiWrapper(temperature=0.7)

        # Basic generation
        print("1. Basic Generation")
        print("-" * 70)
        response = llm.generate("What is Python in one sentence?")
        print(f"Q: What is Python in one sentence?")
        print(f"A: {response}\n")

        # With persona
        print("2. With Persona")
        print("-" * 70)
        llm.set_persona(
            "You are a helpful teacher who explains concepts using simple analogies."
        )
        response = llm.generate("What is machine learning?")
        print(f"Q: What is machine learning?")
        print(f"A: {response}\n")

        # Chat mode
        print("3. Chat Mode (Multi-turn)")
        print("-" * 70)
        print("User: My favorite color is blue")
        r1 = llm.chat("My favorite color is blue")
        print(f"AI: {r1}\n")

        print("User: What's my favorite color?")
        r2 = llm.chat("What's my favorite color?")
        print(f"AI: {r2}\n")

        # Stats
        print("4. Statistics")
        print("-" * 70)
        stats = llm.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\n" + "="*70)
        print("✅ Demo completed successfully!")
        print("="*70 + "\n")

    except ValueError as e:
        print(f"\n❌ Error: {e}\n")
        print("Setup Instructions:")
        print("1. Get API key: https://makersuite.google.com/app/apikey")
        print("2. Create .env file with: GEMINI_API_KEY=your_key_here")
        print("3. Run again\n")


if __name__ == "__main__":
    demo()