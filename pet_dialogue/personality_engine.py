#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Personality Engine module for Pet Dialogue.
Handles different personality traits for dialogue generation.
"""

import random
from typing import Dict, List, Any, Optional
from loguru import logger

class PersonalityEngine:
    """
    Manages different personality traits for dialogue generation.
    
    Provides prompt templates and lexicons for different personality traits:
    - Auto-adapt: Automatically adapts to the animal's behavior
    - Enthusiastic: Excited, high-energy, positive
    - Sassy: Witty, somewhat sarcastic, confident
    - Dramatic: Exaggerated emotions, theatrical
    - Philosophical: Deep thoughts, contemplative
    - Sarcastic: Dry humor, ironic
    """
    
    def __init__(self):
        """Initialize the personality engine."""
        logger.info("Initializing PersonalityEngine")
        
        # Define personality traits and their characteristics
        self.personalities = {
            "auto": {
                "description": "Adapts to the animal's behavior",
                "behaviors": {
                    "excited": "enthusiastic",
                    "anxious": "dramatic",
                    "curious": "philosophical",
                    "playful": "enthusiastic",
                    "relaxed": "philosophical",
                    "alert": "sassy"
                }
            },
            "enthusiastic": {
                "description": "Excited, high-energy, positive",
                "lexicon": [
                    "AMAZING", "OMG", "WOW", "BEST", "LOVE", "SUPER",
                    "AWESOME", "INCREDIBLE", "ABSOLUTELY", "SO MUCH"
                ],
                "templates": [
                    "This is the BEST THING EVER! {action}!",
                    "OH MY GOSH! {action}!",
                    "I CAN'T EVEN! {action}!",
                    "This is AMAZING! {action}!",
                    "WOOHOOO! {action}!"
                ]
            },
            "sassy": {
                "description": "Witty, somewhat sarcastic, confident",
                "lexicon": [
                    "excuse me", "obviously", "please", "um", "hello",
                    "seriously", "whatever", "actually", "frankly", "honestly"
                ],
                "templates": [
                    "Excuse me? {action}. Deal with it.",
                    "Um, {action}. As if you didn't know.",
                    "Let's be clear: {action}. And I look fabulous doing it.",
                    "Oh honey, {action}. It's called having standards.",
                    "{action}. Not that I care what you think."
                ]
            },
            "dramatic": {
                "description": "Exaggerated emotions, theatrical",
                "lexicon": [
                    "absolutely", "desperately", "tragically", "magnificently",
                    "devastatingly", "gloriously", "catastrophically", "dramatically",
                    "shockingly", "astoundingly"
                ],
                "templates": [
                    "The DRAMA! The SUSPENSE! {action}!",
                    "My ENTIRE EXISTENCE has led to this moment where {action}!",
                    "I simply CANNOT BELIEVE that {action}! The AUDACITY!",
                    "BEHOLD! {action}! WITNESS ME!",
                    "This is the GREATEST CHALLENGE of my life! {action}!"
                ]
            },
            "philosophical": {
                "description": "Deep thoughts, contemplative",
                "lexicon": [
                    "contemplating", "pondering", "considering", "reflecting",
                    "perceiving", "observing", "questioning", "analyzing",
                    "examining", "wondering"
                ],
                "templates": [
                    "One must wonder... is {action} truly the meaning of existence?",
                    "In the grand tapestry of life, {action} is but a single thread.",
                    "What is {action} if not the universe experiencing itself?",
                    "To {action} or not to {action}, that is the eternal question.",
                    "I contemplate: {action}. What does it all mean?"
                ]
            },
            "sarcastic": {
                "description": "Dry humor, ironic",
                "lexicon": [
                    "obviously", "clearly", "totally", "absolutely", "supposedly",
                    "apparently", "allegedly", "presumably", "wow", "sure"
                ],
                "templates": [
                    "Oh great, {action}. Just what I always wanted.",
                    "Sure, {action}. Because my life wasn't complicated enough.",
                    "Wow, {action}. I'm absolutely thrilled. Can you tell?",
                    "Yeah, {action}. That makes perfect sense. Not.",
                    "Oh look at me, {action}. Isn't that just special."
                ]
            }
        }
        
        logger.info("PersonalityEngine initialized")
    
    def get_prompt_template(self, personality: str, animal_type: str, behavior: str) -> str:
        """
        Get a prompt template for the specified personality.
        
        Args:
            personality: Personality trait
            animal_type: Type of animal ("dog" or "cat")
            behavior: Behavior state
            
        Returns:
            Prompt template string
        """
        # Auto-adapt personality based on behavior
        if personality == "auto":
            matched_personality = self.personalities["auto"]["behaviors"].get(behavior, "enthusiastic")
        else:
            matched_personality = personality
        
        # Fallback to enthusiastic if personality not found
        if matched_personality not in self.personalities:
            matched_personality = "enthusiastic"
        
        # Get personality description
        personality_desc = self.personalities[matched_personality]["description"]
        
        return f"""Act as a {matched_personality} {animal_type} who is feeling {behavior}.
Generate a single short, humorous sentence in first-person perspective that captures your current state of mind.
Use colloquial language appropriate for a {animal_type} with a {personality_desc} personality.
Focus on your immediate experience, surroundings, or what you're thinking about right now.
Don't use emojis and keep it family-friendly.

Your thought should relate to: {behavior}

For example:
- {matched_personality} dog feeling excited: "{self._get_example(matched_personality, 'dog', 'excited')}"
- {matched_personality} cat feeling curious: "{self._get_example(matched_personality, 'cat', 'curious')}"

Your {animal_type} thought:"""
    
    def enhance_dialogue(self, dialogue: str, personality: str, behavior: str) -> str:
        """
        Enhance a dialogue string with personality-specific traits.
        
        Used when we have a basic dialogue (e.g., from fallback templates)
        and want to add personality-specific traits.
        
        Args:
            dialogue: Base dialogue string
            personality: Personality trait
            behavior: Behavior state
            
        Returns:
            Enhanced dialogue string
        """
        # Auto-adapt personality based on behavior
        if personality == "auto":
            personality = self.personalities["auto"]["behaviors"].get(behavior, "enthusiastic")
        
        # If personality not found or dialogue is empty, return as is
        if personality not in self.personalities or not dialogue:
            return dialogue
        
        # Get personality lexicon
        lexicon = self.personalities[personality].get("lexicon", [])
        
        # 30% chance to enhance with lexicon terms
        if lexicon and random.random() < 0.3:
            # Select a random term from the lexicon
            term = random.choice(lexicon)
            
            # Different enhancement strategies based on personality
            if personality in ["enthusiastic", "dramatic"]:
                # For enthusiastic/dramatic, emphasize with uppercase and exclamation
                dialogue = dialogue.replace(".", "!")
                if random.random() < 0.5:
                    dialogue = dialogue.replace(
                        random.choice(dialogue.split()),
                        term.upper()
                    )
                else:
                    dialogue = f"{term.upper()}! {dialogue}"
            
            elif personality == "philosophical":
                # For philosophical, add contemplative phrases
                if random.random() < 0.5:
                    dialogue = f"I find myself {term}... {dialogue}"
                else:
                    dialogue = f"{dialogue} One must {term}."
            
            elif personality == "sarcastic":
                # For sarcastic, add sarcastic openers or endings
                if random.random() < 0.5:
                    dialogue = f"{term}, {dialogue}"
                else:
                    dialogue = f"{dialogue} {term.capitalize()}..."
            
            elif personality == "sassy":
                # For sassy, add attitude
                if random.random() < 0.5:
                    dialogue = f"{term}, {dialogue}"
                else:
                    dialogue = f"{dialogue} {term.capitalize()} not."
        
        return dialogue
    
    def _get_example(self, personality: str, animal_type: str, behavior: str) -> str:
        """
        Get an example dialogue for the specified parameters.
        
        Args:
            personality: Personality trait
            animal_type: Type of animal
            behavior: Behavior state
            
        Returns:
            Example dialogue string
        """
        # Get templates for the personality
        templates = self.personalities.get(personality, {}).get("templates", [])
        
        # If no templates, return a generic example
        if not templates:
            if animal_type == "dog":
                return "I'm wagging my tail because I'm super excited to see you!"
            else:
                return "I'm watching this mysterious red dot with intense focus."
        
        # Select a random template
        template = random.choice(templates)
        
        # Fill in action based on animal type and behavior
        actions = {
            "dog": {
                "excited": "my tail is wagging at supersonic speeds",
                "anxious": "that loud noise is definitely a threat to our safety",
                "curious": "I need to investigate that strange smell immediately",
                "playful": "this toy is the most important thing in the universe right now",
                "relaxed": "this sunny spot is perfectly positioned for my nap",
                "alert": "something moved in the bushes and I must alert the entire neighborhood"
            },
            "cat": {
                "excited": "this box is the greatest discovery in feline history",
                "anxious": "that vacuum cleaner is clearly plotting against me",
                "curious": "I must know what's behind this closed door",
                "playful": "this paper bag has challenged me to a duel",
                "relaxed": "I have claimed this sunbeam as my personal property",
                "alert": "an unknown entity has entered my territory"
            }
        }
        
        # Get action for animal type and behavior
        action = actions.get(animal_type, {}).get(behavior, "I'm doing something very interesting")
        
        # Return filled template
        return template.format(action=action) 