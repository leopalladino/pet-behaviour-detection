#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialogue Generator module for Pet Dialogue.
Generates dialogue based on animal behavior.
"""

import os
import random
import json
import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional
from loguru import logger

from pet_dialogue.utils import load_json_data, get_project_root
from pet_dialogue.personality_engine import PersonalityEngine

class DialogueGenerator:
    """
    Generates dialogue based on detected animal behavior.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the dialogue generator.
        
        Args:
            model_path: Path to local model directory (defaults to a small model if not specified)
        """
        logger.info("Initializing DialogueGenerator with local model")
        
        # Set model path
        self.model_path = model_path or os.environ.get("LOCAL_MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.use_local_model = True
        
        # Flag to avoid loading the model until needed (lazy loading)
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Load dialogue templates
        templates_path = os.path.join(get_project_root(), "pet_dialogue", "data", "dialogue_templates.json")
        self.templates = load_json_data(templates_path)
        
        # Initialize personality engine
        self.personality_engine = PersonalityEngine()
        
        logger.info(f"DialogueGenerator initialized with local model: {self.model_path}")
    
    async def generate(self, behavior: Dict[str, Any], personality: str = "auto") -> str:
        """
        Generate dialogue based on animal behavior and personality.
        
        Args:
            behavior: Behavior classification results
            personality: Personality trait for dialogue generation
            
        Returns:
            Generated dialogue string
        """
        # Extract behavior details
        animal_type = behavior.get("animal_type", "dog").lower()
        state = behavior.get("state", "relaxed").lower()
        confidence = behavior.get("confidence", 0.0)
        
        logger.info(f"Generating dialogue for {animal_type} with {state} state (confidence: {confidence:.2f})")
        
        # Handle stress/distress with ethical guardrails but allow angry/irritated states
        if state in ["stressed", "fearful", "pain"]:
            # Ethical guardrail: don't anthropomorphize serious distress signals
            logger.warning(f"Detected potential distress state: {state}. Using generic dialogue.")
            return self._create_dynamic_dialogue(animal_type, "relaxed", personality)
        
        # Map "aggressive" or similar states to "angry" for dialogue generation
        if state in ["aggressive", "angry", "irritated", "annoyed"]:
            state = "angry"  # Normalize to a single angry state
            logger.info(f"Mapped state to 'angry' for dialogue generation")
        
        # Try local model first
        try:
            if self.use_local_model:
                dialogue = await self._generate_with_local_model(animal_type, state, personality)
                if dialogue and len(dialogue) > 5:  # Make sure we got something substantial
                    return dialogue
        except Exception as e:
            logger.warning(f"Local model generation failed: {str(e)}. Using dynamic dialogue.")
        
        # Always fall back to dynamic dialogue generation if model fails or returns empty
        return self._create_dynamic_dialogue(animal_type, state, personality)
    
    async def _generate_with_local_model(self, animal_type: str, state: str, personality: str) -> str:
        """
        Generate dialogue using a local model.
        
        Args:
            animal_type: Type of animal ("dog" or "cat")
            state: Behavior state
            personality: Personality trait
            
        Returns:
            Generated dialogue string
        """
        # Get the appropriate prompt template based on personality
        prompt = self.personality_engine.get_prompt_template(personality, animal_type, state)
        
        # Load model on first use
        if not self.model_loaded:
            try:
                await self._load_model()
            except Exception as e:
                logger.error(f"Failed to load local model: {str(e)}")
                return ""  # Return empty string to trigger fallback
        
        # Generate text with the model
        try:
            # Run inference - wrap in asyncio.to_thread to make it non-blocking
            generated_text = await asyncio.to_thread(self._run_inference, prompt)
            
            # Clean up the response
            if generated_text:
                # Remove any prefixes like "A dog's thought:" that the model might add
                lines = generated_text.split("\n")
                for line in lines:
                    cleaned = line.strip()
                    if cleaned and not cleaned.startswith(("Your", "The", "A ", "An ", "-", "*")):
                        return cleaned
            
            return ""
        except Exception as e:
            logger.warning(f"Local model inference failed: {str(e)}")
            return ""
    
    async def _load_model(self):
        """Load the local model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model with reduced precision for efficiency but without using device_map or low_cpu_mem_usage
            # to avoid requiring accelerate
            precision = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=precision
            )
            
            # Manually move to CUDA if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            self.model_loaded = False
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _run_inference(self, prompt: str) -> str:
        """
        Run inference with the local model.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated text
        """
        try:
            # Import torch here as well in case it's lost in the async context
            import torch
            
            # Encode prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and return
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response = generated_text[len(prompt):].strip()
            
            # Clean up the response - this is where we'll improve filtering
            cleaned_response = self._clean_model_output(response)
            return cleaned_response
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return ""
    
    def _clean_model_output(self, text: str) -> str:
        """
        Clean model output to ensure only dialogue is returned.
        Removes meta instructions, notes, and other non-dialogue text.
        
        Args:
            text: Raw model output text
            
        Returns:
            Cleaned dialogue text
        """
        # If empty or too short, return empty to trigger fallback
        if not text or len(text) < 3:
            return ""
            
        # Split by lines to process each separately
        lines = text.split("\n")
        
        # Look for actual dialogue lines
        for line in lines:
            # Skip empty lines
            line = line.strip()
            if not line:
                continue
                
            # Skip lines that are clearly meta-instructions
            if any(keyword in line.lower() for keyword in ["note:", "instruction", "example", "prompt", "output:", "context:", "personality:", "tone:", "humorous", "colloquial"]):
                continue
                
            # Skip lines that start with common prefixes
            if line.startswith(("Your", "The", "-", "*", "•", "1.", "2.", "#")):
                continue
                
            # Skip any line with more than one colon (likely formatting instructions)
            if line.count(":") > 1:
                continue
                
            # If we have an acceptable line with reasonable dialogue length, return it
            if 5 <= len(line) <= 100:
                return line
        
        # If we couldn't find good dialogue, fall back to dynamic generation
        return ""
    
    def _create_dynamic_dialogue(self, animal_type: str, state: str, personality: str) -> str:
        """
        Create a dynamic dialogue without using templates.
        
        Args:
            animal_type: Type of animal ("dog" or "cat")
            state: Behavior state
            personality: Personality trait
            
        Returns:
            Generated dialogue string
        """
        # Expanded dynamic dialogue parts with more variety
        dog_thoughts = {
            "relaxed": [
                "This spot is perfect for a nap.",
                "I could stay here all day long.",
                "Just enjoying the moment right now.",
                "Nothing better than some quiet time.",
                "I'm at peace with the universe right now.",
                "This is what dog heaven feels like.",
                "I've achieved peak relaxation right now.",
                "I'm in my happy place.",
                "So comfortable I might never move again.",
                "Pure contentment, that's what this is.",
                "My human gives the best spots to rest.",
                "I deserve this break after all my hard work barking.",
                "Life is good when you can just be still."
            ],
            "playful": [
                "Let's play! Right now! Come on!",
                "I feel like zooming around!",
                "This is so much fun!",
                "I could play all day long!",
                "Chase me! Or I'll chase you!",
                "Throw the ball! Throw it again!",
                "Play is the best part of being a dog!",
                "My tail can't wag fast enough!",
                "Bounce! Jump! Spin! Play!",
                "I'm in full play mode and nothing can stop me!",
                "Who needs naps when you can play?",
                "I feel the zoomies coming on!",
                "Play time is the best time!"
            ],
            "alert": [
                "Did you hear that?",
                "Something's happening outside.",
                "I need to check that out.",
                "That sound needs investigation.",
                "I'm watching everything very carefully.",
                "My ears are picking up something.",
                "There's a disturbance that requires my attention.",
                "Security scan in progress. Stay alert.",
                "I sense a potential threat... or maybe a squirrel.",
                "My guard dog instincts are tingling.",
                "Someone might be approaching our territory.",
                "That noise is suspicious... I better pay attention."
            ],
            "curious": [
                "What's that over there?",
                "I wonder what this is...",
                "I need to explore that right now.",
                "That's new and interesting!",
                "I've never seen that before!",
                "Must investigate this strange thing.",
                "My nose needs to know what this is.",
                "This smell is fascinating and new.",
                "I'm picking up an intriguing scent.",
                "Something unfamiliar is happening and I need details.",
                "My investigation skills are needed here.",
                "That's different from yesterday. I need to inspect it."
            ],
            "angry": [
                "Back off right now. I mean it!",
                "This is MY territory. Don't test me.",
                "I'm not in the mood for this.",
                "You're really pushing my buttons right now.",
                "One more step and you'll regret it.",
                "Do I look like I want to be messed with?",
                "I've had it up to HERE with this nonsense.",
                "This is your final warning. Stand down.",
                "My patience has officially run out.",
                "I'm showing my teeth for a reason.",
                "I've been pushed too far this time.",
                "You're about to see a side of me you won't like."
            ],
            "excited": [
                "OMG OMG OMG! Is that a SQUIRREL?!",
                "HELLO HELLO HELLO! I love you more than I did 5 seconds ago!",
                "My tail is going so fast it might achieve liftoff! Woooooo!",
                "BALL! BALL! BALL! Nothing else in the universe matters right now!",
                "Walkies?! Did someone say WALKIES?! Best. Day. EVER!",
                "You're HOME! This is literally the greatest moment of my life... again!",
                "I'm so excited I might explode into a million happy pieces!",
                "The door! Someone's at the door! MUST ANNOUNCE WITH MAXIMUM ENTHUSIASM!",
                "Treat? Treat? Did the treat bag just make a noise? I HEARD IT!",
                "I can't contain my excitement! My tail is a blur!",
                "This is the BEST THING EVER! I need to tell everyone!",
                "I'm so happy I might do a backflip! Or at least try to!"
            ]
        }
        
        cat_thoughts = {
            "relaxed": [
                "Perfect spot for my royal nap.",
                "I have achieved optimal comfort.",
                "Contemplating my kingdom now.",
                "Humans exist to serve my relaxation.",
                "This is my favorite sunbeam.",
                "I've found the warmest spot in the house.",
                "Peak cat comfort has been achieved.",
                "I shall now rest my majestic self.",
                "My dignity requires at least 16 hours of rest daily.",
                "I am one with this comfortable surface.",
                "Being this serene is my full-time occupation.",
                "I've mastered the art of doing absolutely nothing."
            ],
            "playful": [
                "That thing must be hunted!",
                "I'm feeling particularly nimble right now.",
                "Time to pounce on something!",
                "My hunting skills need practice.",
                "I feel the zoomies coming on.",
                "My wild instincts have taken over!",
                "Watch my superior hunting technique!",
                "This toy doesn't stand a chance against me.",
                "My predator mode is fully activated.",
                "I must attack this immediately!",
                "My reflexes are lightning fast right now.",
                "The hunt is on and I am unstoppable!"
            ],
            "alert": [
                "Something moved. I must observe.",
                "That noise requires investigation.",
                "I detect a disturbance in my domain.",
                "My senses are telling me something's up.",
                "I'm tracking something interesting.",
                "My whiskers sense a change in the air.",
                "Someone unfamiliar is approaching.",
                "I must remain vigilant at all times.",
                "Potential threat detected. Analyzing.",
                "Everything in my territory must be monitored.",
                "My ears are detecting unusual sounds.",
                "A disturbance requires my full attention now."
            ],
            "curious": [
                "What's in this box?",
                "That thing needs thorough inspection.",
                "I must investigate this new object.",
                "My whiskers sense something unfamiliar.",
                "This requires my immediate attention.",
                "I need to determine if this is worth my time.",
                "My natural curiosity demands exploration.",
                "I must sniff this thoroughly to understand it.",
                "This item wasn't here before. Suspicious.",
                "I shall examine this from every angle.",
                "My paws need to touch this mysterious object.",
                "New things in my territory must be assessed properly."
            ],
            "angry": [
                "Touch my stuff again. I dare you.",
                "I've had enough of this treatment.",
                "You're on thin ice right now.",
                "This tail flicking is not a good sign for you.",
                "My ears are back for a reason.",
                "One more disturbance and there will be consequences.",
                "I'm officially irritated with this situation.",
                "This is completely unacceptable behavior.",
                "My patience has limits, and you've found them.",
                "You're about to see why cats are apex predators.",
                "I'm done tolerating this nonsense.",
                "Cross me again at your own peril."
            ],
            "excited": [
                "It's 3 AM. Time for my championship sprinting event across your face!",
                "I've been waiting ALL DAY to show you this dead bug I found!",
                "The red dot has returned! This time I WILL catch it!",
                "I can see the bottom of my food bowl. This is a CATASTROPHE!",
                "Fresh catnip! Time to lose all my dignity!",
                "A box! A NEW box! This is the greatest day of my nine lives!",
                "The bird is back at the window! This is not a drill, people!",
                "My favorite toy was under the couch this WHOLE TIME!",
                "The curtains are dancing in the breeze. They're taunting me. I must attack!",
                "I'm so excited I might do a backflip! Or at least try to!",
                "The treat jar just made a noise! This is the moment I've been waiting for!",
                "I can't contain my excitement! My tail is a blur!"
            ]
        }
        
        # Get appropriate thoughts based on animal type
        thoughts = dog_thoughts if animal_type == "dog" else cat_thoughts
        
        # Default to relaxed if state not found
        if state not in thoughts:
            logger.warning(f"No dialogue templates found for state '{state}'. Defaulting to relaxed.")
            state = "relaxed"
        
        # Get a random thought for the state - make sure we don't pick recently used ones
        # Using a deterministic but varied approach instead of simple random
        
        # Use the current time to create a hash seed to ensure variety
        time_seed = str(int(time.time() / 10))  # Changes every 10 seconds
        hash_object = hashlib.md5((animal_type + state + time_seed).encode())
        hash_value = int(hash_object.hexdigest(), 16)
        
        # Select a phrase using the hash value for better distribution
        available_thoughts = thoughts[state]
        selected_index = hash_value % len(available_thoughts)
        base_thought = available_thoughts[selected_index]
        
        # Enhance based on personality with more variety in the modifiers
        if personality == "enthusiastic":
            modifiers = ["!!!", "! This is amazing!", "! I'm so excited!", "! Best thing ever!"]
            # For angry states, use appropriate modifiers
            if state == "angry":
                modifiers = ["!!!", "! I mean it!", "! I'm serious!", "! No doubt about it!"]
            modifier = modifiers[hash_value % len(modifiers)]
            return f"{base_thought}{modifier}"
        elif personality == "sassy":
            modifiers = [" Obviously.", " Duh.", " As if you didn't know.", " I mean, come on."]
            # For angry states, use appropriate modifiers
            if state == "angry":
                modifiers = [" And I'm not kidding.", " Try me.", " You've been warned.", " Believe it."]
            modifier = modifiers[hash_value % len(modifiers)]
            return f"{base_thought}{modifier}"
        elif personality == "dramatic":
            modifiers = ["Oh my goodness!", "I can't even!", "You won't believe this!", "This is absolutely incredible!"]
            # For angry states, use appropriate modifiers
            if state == "angry":
                modifiers = ["I am OUTRAGED!", "This is UNBELIEVABLE!", "I cannot EXPRESS my anger!", "The AUDACITY!"]
            modifier = modifiers[hash_value % len(modifiers)]
            return f"{modifier} {base_thought}"
        elif personality == "philosophical":
            modifiers = ["I wonder...", "One must ponder...", "It occurs to me that...", "In the grand scheme of things..."]
            # For angry states, use appropriate modifiers
            if state == "angry":
                modifiers = ["It is with great contemplation that I declare...", "After much reflection, I must say...", 
                            "The universe has led me to this conclusion:", "In my wisdom, I must assert:"]
            modifier = modifiers[hash_value % len(modifiers)]
            return f"{modifier} {base_thought}"
        elif personality == "sarcastic":
            modifiers = ["Yeah, right.", "Oh sure.", "As if.", "Totally thrilling:"]
            # For angry states, use appropriate modifiers too
            if state == "angry":
                modifiers = ["Oh perfect.", "Just great.", "Wonderful.", "Exactly what I needed:"]
            modifier = modifiers[hash_value % len(modifiers)]
            return f"{modifier} {base_thought}"
        else:
            # For "auto" or other personalities, just return the base thought
            return base_thought
    
    def _generate_from_templates(self, animal_type: str, state: str, personality: str) -> str:
        """
        Generate dialogue from templates (NOT USED ANYMORE - FALLBACK ONLY).
        
        Args:
            animal_type: Type of animal ("dog" or "cat")
            state: Behavior state
            personality: Personality trait
            
        Returns:
            Generated dialogue string
        """
        # Just redirect to dynamic dialogue generation
        return self._create_dynamic_dialogue(animal_type, state, personality)
    
    def _get_generic_dialogue(self, animal_type: str) -> str:
        """
        Get a generic dialogue for when specific templates are not available.
        
        Args:
            animal_type: Type of animal ("dog" or "cat")
            
        Returns:
            Generic dialogue string
        """
        # More varied and natural-sounding generic responses
        dog_generic = [
            "I'm just being a dog, doing dog things.",
            "What an interesting moment to be a dog.",
            "So many smells, so little time.",
            "Just thinking about dog business.",
            "Life is good when you're a dog."
        ]
        
        cat_generic = [
            "Cats rule, everyone else drools.",
            "Just cat things happening here.",
            "Plotting world domination, as usual.",
            "Observing my domain with feline grace.",
            "Humans are so amusing to watch."
        ]
        
        dialogues = dog_generic if animal_type == "dog" else cat_generic
        return random.choice(dialogues) 