import gradio as gr
import torch
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer
import random
import os
import sys
import time
from threading import Thread

# Set PyTorch MPS fallback for Apple Silicon compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Check for dev mode
DEV_MODE = "--dev" in sys.argv

# Import spaces for HuggingFace deployment
try:
    import spaces
    HF_SPACES = True
    print("🚀 Running on HuggingFace Spaces with ZeroGPU")
except ImportError:
    HF_SPACES = False
    print("🏠 Running locally - spaces module not available")

# MCP is always enabled
print("🔌 MCP protocol enabled - tools available for external access")

MAX_SEED = 2**32 - 1

# Liste des catégories pour le chatbot (100+ catégories variées)
CHAT_CATEGORIES = [
    # Animaux et créatures
    "animal", "bird", "sea creature", "insect", "mythical creature", "prehistoric creature",
    
    # Couleurs et teintes  
    "color", "shade", "metallic color", "gemstone color",
    
    # Objets et artefacts
    "weapon", "tool", "musical instrument", "piece of furniture", "ancient artifact", "modern gadget",
    
    # Émotions et traits
    "emotion", "personality trait", "mood", "mental state", "virtue", "flaw",
    
    # Nature et éléments
    "natural element", "weather phenomenon", "season", "time of day", "celestial body", "landscape",
    
    # Plantes et végétaux
    "flower", "tree", "herb", "fruit", "vegetable", "mushroom",
    
    # Arts et culture
    "art style", "musical genre", "dance style", "literary genre", "architectural style", "fashion style",
    
    # Matériaux et textures
    "fabric", "metal", "stone", "wood type", "crystal", "texture",
    
    # Géographie et lieux
    "country", "city type", "building", "room", "natural landmark", "climate zone",
    
    # Histoire et époques
    "historical period", "ancient civilization", "mythology", "legend", "cultural tradition",
    
    # Sciences et cosmos
    "planet", "star type", "galaxy", "chemical element", "geometric shape", "mathematical concept",
    
    # Sens et perceptions
    "scent", "taste", "sound", "touch sensation", "visual effect", "temperature",
    
    # Énergies et forces
    "type of energy", "natural force", "magical power", "spiritual element", "life force",
    
    # Professions et rôles
    "profession", "fantasy role", "mythical being", "guardian spirit", "mentor figure",
    
    # Activités et actions
    "hobby", "sport", "art form", "ritual", "celebration", "journey type",
    
    # Abstractions et concepts
    "philosophical concept", "virtue", "sin", "dream", "fear", "hope", "memory type",
    
    # Objets magiques et fantastiques
    "magical item", "enchanted object", "potion ingredient", "spell component", "rune",
    
    # Environnements spéciaux
    "mystical place", "hidden realm", "sacred space", "forbidden zone", "lost city"
]

# Variable globale pour tracking des catégories utilisées dans la conversation
used_categories = []

def get_next_category():
    """Retourne une catégorie aléatoire non utilisée"""
    available_categories = [cat for cat in CHAT_CATEGORIES if cat not in used_categories]
    
    if not available_categories:
        # Si toutes les catégories ont été utilisées, reset
        used_categories.clear()
        available_categories = CHAT_CATEGORIES.copy()
    
    category = random.choice(available_categories)
    used_categories.append(category)
    print(f"🎲 DEBUG Selected category: {category} (used: {len(used_categories)}/{len(CHAT_CATEGORIES)})")
    return category

def simple_chat_response(user_message, history):
    """Logique simple de chat sans LLM - pose juste la question suivante"""
    if not user_message.strip():
        return "Please type your answer."
    
    # Si c'est le début de la conversation
    if len(history) == 0 or user_message.lower().strip() in ["ready", "start", "begin"]:
        used_categories.clear()  # Reset les catégories
        category = get_next_category()
        return f"If you were {category}, what would you be?"
    
    # Sinon, poser la question suivante
    category = get_next_category()
    return f"If you were {category}, what would you be?"

def load_flux_model():
    dtype = torch.bfloat16
    
    # For HuggingFace Spaces, prioritize CUDA
    if HF_SPACES and torch.cuda.is_available():
        device = "cuda"
    # For local development, prioritize MPS for Apple Silicon
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device for FLUX: {device}")
    
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", 
        torch_dtype=dtype
    ).to(device)
    
    # Print tokenizer info for debugging
    if hasattr(pipe, 'tokenizer'):
        print(f"FLUX Tokenizer max length: {pipe.tokenizer.model_max_length}")
    if hasattr(pipe, 'tokenizer_2'):
        print(f"FLUX Tokenizer 2 max length: {pipe.tokenizer_2.model_max_length}")
    
    return pipe

def load_gemma_model():
    print("Loading Gemma-3n-E2B-it model...")
    
    model_id = "google/gemma-3n-E2B-it"
    processor = AutoProcessor.from_pretrained(model_id)
    
    if HF_SPACES:
        # Don't load model in main process for ZeroGPU
        print("ZeroGPU mode: Model will be loaded in GPU functions")
        return processor, None
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        print(f"Using device for Gemma-E2B: {model.device}")
        return processor, model

flux_pipe = load_flux_model()
gemma_processor, gemma_model = load_gemma_model()

# Using simplified chat system for conversation flow

def extract_between_balises(s):
    """Extract content between ===BEGIN=== and ===END=== balises and optimize for FLUX"""
    import re
    print(f"🔍 DEBUG extract_between_balises input: '{s}'")
    
    # Utilise DOTALL pour inclure tous les caractères, même les retours à la ligne
    m = re.search(r'===BEGIN===([\s\S]*?)===END===', s, flags=re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        print(f"🔍 DEBUG found between balises: '{content}'")
        
        # Séparer les critères (par virgule ou ligne)
        if ',' in content:
            criteria = [c.strip() for c in content.split(',') if c.strip()]
        else:
            criteria = [line.strip() for line in content.splitlines() if line.strip()]
        
        print(f"🔍 DEBUG criteria found: {criteria}")
        
        # Construire le prompt en ajoutant les critères un par un sans dépasser 80 caractères
        base_prompt = "digital portrait"
        current_prompt = base_prompt
        
        for criterion in criteria:
            # Test si on peut ajouter ce critère
            test_prompt = f"{current_prompt}, {criterion}" if current_prompt != base_prompt else f"{current_prompt}: {criterion}"
            
            if len(test_prompt) <= 80:
                current_prompt = test_prompt
                print(f"🔍 DEBUG added criterion: '{criterion}' -> '{current_prompt}' ({len(current_prompt)} chars)")
            else:
                print(f"🔍 DEBUG skipping criterion (would exceed 80 chars): '{criterion}'")
                break
        
        print(f"🔍 DEBUG final prompt: '{current_prompt}' ({len(current_prompt)} chars)")
        return current_prompt
    
    print(f"🔍 DEBUG balises not found, using fallback")
    # Fallback : construire un prompt basique
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if lines:
        # Prendre la ligne la plus courte qui pourrait être un bon prompt
        short_lines = [line for line in lines if len(line) < 60]
        if short_lines:
            result = f"digital portrait: {short_lines[0]}"
        else:
            result = f"digital portrait: {lines[0][:60]}"
        print(f"🔍 DEBUG fallback result: '{result}'")
        return result
    
    print(f"🔍 DEBUG returning default")
    return "digital portrait: artistic avatar"

@spaces.GPU() if HF_SPACES else lambda x: x
def generate_flux_prompt_from_description(original_prompt):
    """Use Gemma model to generate optimized FLUX prompt from description"""
    print(f"🔍 DEBUG original_prompt input: '{original_prompt}'")
    
    # Instruction ultra-simple pour Gemma
    instruction = f"""List visual elements from: {original_prompt}

Elements:"""

    print(f"🔍 DEBUG instruction length: {len(instruction)} chars")
    print(f"🔍 DEBUG instruction: '{instruction}'")

    try:
        # Use Gemma model for text generation
        model = gemma_model if gemma_model is not None else _load_gemma_model_gpu()
        processor = gemma_processor
        
        # Use tokenizer directly for text input
        input_ids = processor.tokenizer(instruction, return_tensors="pt")
        if hasattr(model, 'device'):
            input_ids = {k: v.to(model.device) for k, v in input_ids.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=80,  # Court et précis
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.2,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        input_length = input_ids['input_ids'].shape[1]
        generated_text = processor.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        print(f"🔍 DEBUG Gemma generated output: '{generated_text}'")

        # Parse output more intelligently - stop at first explanation or verbose content
        lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
        
        key_elements = []
        for line in lines[:8]:  # Check more lines but filter better
            # Stop at explanatory sections
            if any(stop_word in line.lower() for stop_word in ['here\'s', 'explanation', 'prompt:', 'inspired by', 'captures', 'why it works']):
                break
                
            # Clean and extract meaningful elements
            cleaned = line.replace('-', '').replace('*', '').replace('•', '').replace(':', '').strip()
            
            # Split by comma if multiple elements in one line
            if ',' in cleaned:
                parts = [p.strip() for p in cleaned.split(',')]
                for part in parts[:3]:  # Max 3 per line
                    if len(part) > 2 and len(part) < 20 and not any(skip in part.lower() for skip in ['based on', 'portrait', 'avatar', 'elements', 'figure', 'detailed']):
                        key_elements.append(part)
            else:
                if len(cleaned) > 2 and len(cleaned) < 20 and not any(skip in cleaned.lower() for skip in ['based on', 'portrait', 'avatar', 'elements', 'figure', 'detailed']):
                    key_elements.append(cleaned)
        
        if key_elements:
            # Build optimized prompt
            result = "digital portrait: " + ", ".join(key_elements[:4])  # Max 4 elements to stay under 80 chars
            if len(result) > 80:
                # Truncate to fit
                result = "digital portrait: " + ", ".join(key_elements[:3])
            print(f"🔍 DEBUG final Gemma result: '{result}' ({len(result)} chars)")
            return result
        else:
            return "digital portrait: artistic avatar"
        
    except Exception as e:
        print(f"🔍 DEBUG Gemma error: {e}")
        # Fallback en cas d'erreur
        return f"digital portrait: artistic avatar based on {original_prompt[:30]}..."

# Model loading function for GPU contexts with caching
_cached_gpu_model = None

def _load_gemma_model_gpu():
    """Load model inside GPU context with caching"""
    global _cached_gpu_model
    if _cached_gpu_model is None:
        print("🔄 Loading Gemma model in GPU context...")
        model_id = "google/gemma-3n-E2B-it"
        _cached_gpu_model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        print("✅ Gemma model loaded and cached")
    else:
        print("♻️ Using cached Gemma model")
    return _cached_gpu_model

# Multilingual support
def get_translations():
    return {
        "en": {
            "title": "🎭 Avatar Generator - Chinese Portrait",
            "subtitle": "Complete at least the first 3 groups to generate your personalized avatar.",
            "portrait_title": "📝 Chinese Portrait (first 3 groups required)",
            "group": "Group",
            "required": "Required",
            "optional": "Optional",
            "if_i_was": "If I was",
            "i_would_be": "I would be",
            "generate_btn": "🎨 Generate Avatar",
            "avatar_title": "🖼️ Generated Avatar",
            "your_avatar": "Your Avatar",
            "information": "Information",
            "error_required": "Error: The first 3 groups of fields are required.",
            "success": "Avatar generated successfully!",
            "prompt_used": "Prompt used:",
            "error_generation": "Error during generation:",
            "footer": "Avatar generated with FLUX.1-schnell",
            "quality_normal": "Normal Quality (4 steps, 512x512)",
            "quality_high": "High Quality (8 steps, 512x512)",
            "quality_label": "Quality:",
            "tab_form": "📝 Form Mode",
            "tab_chat": "💬 Chat Mode",
            "chat_title": "🤖 AI Assistant - Avatar Creator",
            "chat_subtitle": "Let me guide you through creating your Chinese portrait!",
            "thinking": "Thinking...",
            "placeholders": {
                "animal": "an animal...",
                "animal_answer": "a lion...",
                "color": "a color...",
                "color_answer": "red...",
                "object": "an object...",
                "object_answer": "a sword...",
                "feeling": "a feeling...",
                "feeling_answer": "joy...",
                "element": "an element...",
                "element_answer": "fire..."
            }
        },
        "fr": {
            "title": "🎭 Générateur d'Avatar - Portrait Chinois",
            "subtitle": "Complétez au minimum les 3 premiers groupes pour générer votre avatar personnalisé.",
            "portrait_title": "📝 Portrait Chinois (3 premiers groupes obligatoires)",
            "group": "Groupe",
            "required": "Obligatoire",
            "optional": "Optionnel",
            "if_i_was": "Si j'étais",
            "i_would_be": "Je serais",
            "generate_btn": "🎨 Générer l'Avatar",
            "avatar_title": "🖼️ Avatar Généré",
            "your_avatar": "Votre Avatar",
            "information": "Informations",
            "error_required": "Erreur: Les 3 premiers groupes de champs sont obligatoires.",
            "success": "Avatar généré avec succès!",
            "prompt_used": "Prompt utilisé:",
            "error_generation": "Erreur lors de la génération:",
            "footer": "Avatar généré avec FLUX.1-schnell",
            "quality_normal": "Qualité Normale (4 étapes, 512x512)",
            "quality_high": "Haute Qualité (8 étapes, 512x512)",
            "quality_label": "Qualité:",
            "tab_form": "📝 Mode Formulaire",
            "tab_chat": "💬 Mode Chat",
            "chat_title": "🤖 Assistant IA - Créateur d'Avatar",
            "chat_subtitle": "Laissez-moi vous guider pour créer votre portrait chinois!",
            "thinking": "Réflexion...",
            "placeholders": {
                "animal": "un animal...",
                "animal_answer": "un lion...",
                "color": "une couleur...",
                "color_answer": "rouge...",
                "object": "un objet...",
                "object_answer": "une épée...",
                "feeling": "un sentiment...",
                "feeling_answer": "la joie...",
                "element": "un élément...",
                "element_answer": "le feu..."
            }
        }
    }

# Dev mode default values
def get_dev_defaults():
    return {
        "if1": "an animal", "would1": "a majestic wolf",
        "if2": "a color", "would2": "deep purple",
        "if3": "an object", "would3": "an ancient sword",
        "if4": "a feeling", "would4": "fierce determination",
        "if5": "an element", "would5": "lightning"
    }

# Apply ZeroGPU decorator if available
if HF_SPACES:
    @spaces.GPU()
    def generate_avatar(if1: str, would1: str, if2: str, would2: str, if3: str, would3: str, if4: str = "", would4: str = "", if5: str = "", would5: str = "", language: str = "en", quality: str = "normal"):
        """
        Generate a personalized avatar from Chinese portrait elements.
        
        Args:
            if1: First category (e.g., "an animal")
            would1: First answer (e.g., "a majestic wolf")
            if2: Second category (e.g., "a color") 
            would2: Second answer (e.g., "deep purple")
            if3: Third category (e.g., "an object")
            would3: Third answer (e.g., "an ancient sword")
            if4: Fourth category (optional, e.g., "a feeling")
            would4: Fourth answer (optional, e.g., "fierce determination")
            if5: Fifth category (optional, e.g., "an element")
            would5: Fifth answer (optional, e.g., "lightning")
            language: Interface language ("en" or "fr")
            quality: Generation quality ("normal" or "high")
            
        Returns:
            tuple: (generated_image, info_text)
        """
        return _generate_avatar_impl(if1, would1, if2, would2, if3, would3, if4, would4, if5, would5, language, quality)
else:
    def generate_avatar(if1: str, would1: str, if2: str, would2: str, if3: str, would3: str, if4: str = "", would4: str = "", if5: str = "", would5: str = "", language: str = "en", quality: str = "normal"):
        """
        Generate a personalized avatar from Chinese portrait elements.
        
        Args:
            if1: First category (e.g., "an animal")
            would1: First answer (e.g., "a majestic wolf")
            if2: Second category (e.g., "a color") 
            would2: Second answer (e.g., "deep purple")
            if3: Third category (e.g., "an object")
            would3: Third answer (e.g., "an ancient sword")
            if4: Fourth category (optional, e.g., "a feeling")
            would4: Fourth answer (optional, e.g., "fierce determination")
            if5: Fifth category (optional, e.g., "an element")
            would5: Fifth answer (optional, e.g., "lightning")
            language: Interface language ("en" or "fr")
            quality: Generation quality ("normal" or "high")
            
        Returns:
            tuple: (generated_image, info_text)
        """
        return _generate_avatar_impl(if1, would1, if2, would2, if3, would3, if4, would4, if5, would5, language, quality)

@spaces.GPU() if HF_SPACES else lambda x: x
def _generate_avatar_impl(if1, would1, if2, would2, if3, would3, if4, would4, if5, would5, language, quality):
    translations = get_translations()
    t = translations.get(language, translations["en"])
    
    # Validation des champs obligatoires
    if not if1 or not would1 or not if2 or not would2 or not if3 or not would3:
        return None, t["error_required"]
    
    # Construction du prompt style "original_prompt" comme dans testlongflux.py
    portrait_parts = []
    portrait_parts.append(f"If I was {if1} I will be {would1}")
    portrait_parts.append(f"If I was {if2} I will be {would2}")
    portrait_parts.append(f"If I was {if3} I will be {would3}")
    
    if if4 and would4:
        portrait_parts.append(f"If I was {if4} I will be {would4}")
    if if5 and would5:
        portrait_parts.append(f"If I was {if5} I will be {would5}")
    
    # Créer le prompt original style testlongflux.py
    original_prompt = "\n".join(portrait_parts)
    
    # Utiliser le modèle Gemma pour générer le prompt FLUX optimisé
    prompt = generate_flux_prompt_from_description(original_prompt)
    
    try:
        # Configuration selon la qualité
        if quality == "high":
            width, height, steps = 512, 512, 8
        else:
            width, height, steps = 512, 512, 4
            
        # Debug print pour voir le prompt envoyé à FLUX
        print(f"🎨 DEBUG FLUX prompt (Form Mode): '{prompt}'")
        print(f"🎨 DEBUG FLUX prompt length: {len(prompt)} characters")
        
        # Génération avec seed aléatoire
        seed = random.randint(0, MAX_SEED)
        generator = torch.Generator(device=flux_pipe.device).manual_seed(seed)
        
        image = flux_pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator
        ).images[0]
        
        return image, f"{t['success']}\n{t['prompt_used']} {prompt}\nSeed: {seed}\nQuality: {quality} ({steps} steps, {width}x{height})"
    
    except Exception as e:
        return None, f"{t['error_generation']} {str(e)}"

# Separate GPU function for generation only (no generator)
if HF_SPACES:
    @spaces.GPU()
    def gemma_generate_response(message, history, language):
        return _gemma_generate_response_impl(message, history, language)
else:
    def gemma_generate_response(message, history, language):
        return _gemma_generate_response_impl(message, history, language)

# Non-GPU streaming function
def gemma_chat_stream(message, history, language):
    return _gemma_chat_stream_impl(message, history, language)

def _gemma_generate_response_impl(message, history, language):
    """Generate response using GPU - returns complete response"""
    # Load model in GPU context if needed
    model = gemma_model if gemma_model is not None else _load_gemma_model_gpu()
    
    # Prepare messages in the format expected by the processor
    messages = []
    
    # Add history (which already includes the initial system prompt as first user message)
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]})
    
    # Add current message
    messages.append({"role": "user", "content": [{"type": "text", "text": message}]})
    
    # Apply chat template and tokenize
    inputs = gemma_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # Generate response
    with torch.no_grad():
        # Move to device without dtype conversion to avoid issues
        device_inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generate_kwargs = dict(
            device_inputs,
            max_new_tokens=150,
            do_sample=False,
            disable_compile=True,
        )
        
        outputs = model.generate(**generate_kwargs)
        response = gemma_processor.decode(outputs[0][device_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean response
    response = clean_chat_response(response)
    
    return response

def _gemma_chat_stream_impl(message, history, language):
    """Streaming function that calls GPU function and simulates streaming"""
    # For ZeroGPU, we can't use threading with streaming, so fall back to non-streaming
    if HF_SPACES:
        # Get complete response from GPU function
        response = gemma_generate_response(message, history, language)
        
        # Simulate streaming by yielding progressively
        words = response.split()
        partial_response = ""
        updated_history = history + [[message, ""]]
        
        for i, word in enumerate(words):
            partial_response += word + " "
            updated_history[-1][1] = partial_response.strip()
            yield updated_history.copy()
            if i % 3 == 0:  # Pause tous les 3 mots
                time.sleep(0.1)
    else:
        # Local development - use real streaming
        # Load model in GPU context if needed
        model = gemma_model if gemma_model is not None else _load_gemma_model_gpu()
        
        # Prepare messages in the format expected by the processor
        messages = []
        
        # Add history (which already includes the initial system prompt as first user message)
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]})
        
        # Add current message
        messages.append({"role": "user", "content": [{"type": "text", "text": message}]})
        
        # Apply chat template and tokenize
        inputs = gemma_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Set up streaming generation
        streamer = TextIteratorStreamer(gemma_processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        
        # Move to device without dtype conversion to avoid issues
        device_inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generate_kwargs = dict(
            device_inputs,
            streamer=streamer,
            max_new_tokens=150,
            do_sample=False,
            disable_compile=True,
        )
        
        # Generate text in a separate thread
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        # Stream output
        updated_history = history + [[message, ""]]
        output = ""
        try:
            for delta in streamer:
                output += delta
                # Clean the response as it streams
                cleaned_output = clean_chat_response(output)
                updated_history[-1][1] = cleaned_output
                yield updated_history.copy()
        except Exception as e:
            # Fallback to non-streaming if streaming fails
            t.join()  # Wait for thread to complete
            response = gemma_generate_response(message, history, language)
            updated_history[-1][1] = response
            yield updated_history.copy()

def clean_assistant_response(response):
    """
    Nettoie la réponse de l'assistant pour éviter les faux dialogues
    """
    import re
    
    # Enlever les patterns de faux dialogue
    patterns_to_remove = [
        r'User:\s*[^\n]+',  # Enlever "User: ..."
        r'Assistant:\s*[^\n]+',  # Enlever "Assistant: ..."
        r'Human:\s*[^\n]+',  # Enlever "Human: ..."
        r'AI:\s*[^\n]+',    # Enlever "AI: ..."
    ]
    
    cleaned = response.strip()
    
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Enlever les lignes vides multiples
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    # Si la réponse contient encore des patterns de dialogue, couper au premier
    dialogue_patterns = [
        r'(?i)\buser\b.*?:',
        r'(?i)\bassistant\b.*?:',
        r'(?i)\bhuman\b.*?:',
        r'(?i)\bai\b.*?:'
    ]
    
    for pattern in dialogue_patterns:
        match = re.search(pattern, cleaned)
        if match:
            # Couper juste avant le pattern trouvé
            cleaned = cleaned[:match.start()].strip()
            break
    
    # Limiter à 500 caractères max pour éviter les réponses trop longues
    if len(cleaned) > 500:
        # Couper à la dernière phrase complète
        sentences = cleaned[:500].split('.')
        if len(sentences) > 1:
            cleaned = '.'.join(sentences[:-1]) + '.'
        else:
            cleaned = cleaned[:500] + '...'
    
    return cleaned.strip()

def clean_chat_response(response):
    """
    Nettoie la réponse du chat sans limiter la taille autant
    """
    import re
    
    # Enlever les patterns de faux dialogue
    patterns_to_remove = [
        r'User:\s*[^\n]+',  # Enlever "User: ..."
        r'Assistant:\s*[^\n]+',  # Enlever "Assistant: ..."
        r'Human:\s*[^\n]+',  # Enlever "Human: ..."
        r'AI:\s*[^\n]+',    # Enlever "AI: ..."
    ]
    
    cleaned = response.strip()
    
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Enlever les lignes vides multiples
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    # Si la réponse contient encore des patterns de dialogue, couper au premier
    dialogue_patterns = [
        r'(?i)\buser\b.*?:',
        r'(?i)\bassistant\b.*?:',
        r'(?i)\bhuman\b.*?:',
        r'(?i)\bai\b.*?:'
    ]
    
    for pattern in dialogue_patterns:
        match = re.search(pattern, cleaned)
        if match:
            # Couper juste avant le pattern trouvé
            cleaned = cleaned[:match.start()].strip()
            break
    
    # Limiter à 200 caractères max pour le chat (plus généreux que 500 pour l'analyse)
    if len(cleaned) > 200:
        # Couper à la dernière phrase complète
        sentences = cleaned[:200].split('.')
        if len(sentences) > 1:
            cleaned = '.'.join(sentences[:-1]) + '.'
        else:
            cleaned = cleaned[:200] + '...'
    
    return cleaned.strip()

def extract_portrait_from_conversation(history, language="en"):
    """
    Utilise le LLM pour analyser la conversation et synthétiser un prompt d'image dynamique
    """
    # Nettoyer l'historique : enlever le dernier message si c'est une question du modèle sans réponse
    cleaned_history = history.copy()
    if cleaned_history and cleaned_history[-1][1] and not cleaned_history[-1][0]:
        # Si le dernier message a une réponse du modèle mais pas de message utilisateur
        # (dernière entrée est ["", "question_du_modèle"]), on l'enlève
        cleaned_history = cleaned_history[:-1]
    
    # Combiner tout le texte de la conversation nettoyée
    conversation_text = ""
    for user_msg, assistant_msg in cleaned_history:
        if user_msg:  # S'assurer qu'il y a un message utilisateur
            conversation_text += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    
    # Prompt compact pour synthèse directe
    analysis_prompt = f"""Based on the following conversation, generate a compact character description in the style of a Chinese Portrait, formatted as:

Artistic character portrait: [category1] → [answer1], [category2] → [answer2], ...

Only include clear and relevant answers. Skip any incomplete or vague ones.
Do not repeat the full conversation.
Keep the result short (max ~40 tokens), using simple words.

Conversation: {conversation_text}"""

    try:
        # Load model in GPU context if needed
        model = gemma_model if gemma_model is not None else _load_gemma_model_gpu()
        
        # Prepare messages for the new processor format
        messages = [{"role": "user", "content": [{"type": "text", "text": analysis_prompt}]}]
        
        # Apply chat template and tokenize
        inputs = gemma_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Move inputs to device and generate
        with torch.no_grad():
            # Move to device without dtype conversion to avoid issues
            device_inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **device_inputs,
                max_new_tokens=100,
                do_sample=False,
                disable_compile=True,
            )
            
        response = gemma_processor.decode(outputs[0][device_inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Nettoyage léger pour l'analyse (ne pas limiter à 500 caractères)
        response = response.strip()
        # Enlever seulement les patterns de faux dialogue évidents
        import re
        response = re.sub(r'User:\s*[^\n]+', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Assistant:\s*[^\n]+', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Human:\s*[^\n]+', '', response, flags=re.IGNORECASE)
        response = re.sub(r'AI:\s*[^\n]+', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\n\s*\n', '\n', response).strip()
        
        # Extraire le prompt d'image du format attendu
        image_prompt = ""
        
        # Chercher le format "Artistic character portrait:"
        if 'artistic character portrait:' in response.lower():
            # Extraire tout ce qui suit "Artistic character portrait:"
            portrait_index = response.lower().find('artistic character portrait:')
            image_prompt = response[portrait_index:].strip()
        else:
            # Si pas du bon format, prendre la réponse complète et l'ajuster
            image_prompt = response.strip()
            if image_prompt and not image_prompt.lower().startswith('artistic character portrait'):
                image_prompt = f"Artistic character portrait: {image_prompt}"
        
        # S'assurer que le prompt est bien formaté pour FLUX
        if image_prompt:
            # Extraire les éléments pour l'affichage AVANT d'ajouter les éléments artistiques
            elements = []
            if '→' in image_prompt:
                # Parser les éléments au format "category → value"
                parts = image_prompt.split(':')[-1]  # Prendre après ":"
                pairs = parts.split(',')
                for pair in pairs:
                    if '→' in pair and not any(art_word in pair.lower() for art_word in ['high-quality', 'digital art', 'detailed', 'fantasy']):
                        try:
                            category, value = pair.split('→', 1)
                            category = category.strip()
                            value = value.strip().rstrip('.')
                            # Nettoyer la valeur pour enlever les informations d'art
                            value = re.sub(r'\.\s*(high-quality|digital art|fantasy|detailed).*', '', value, flags=re.IGNORECASE).strip()
                            if category and value and not value.startswith('('):
                                elements.append((category, value))
                        except:
                            continue
            
            # Ajouter des éléments artistiques si manquants
            if not any(word in image_prompt.lower() for word in ['detailed', 'high-quality', 'digital art']):
                image_prompt += ". High-quality digital art, fantasy style, detailed illustration"
            
            return image_prompt, elements
        else:
            fallback_prompt = "Artistic character portrait of a unique individual. High-quality digital art, fantasy style, detailed illustration"
            return fallback_prompt, [('style', 'unique individual')]
        
    except Exception as e:
        # Fallback simple en cas d'erreur
        fallback_prompt = "Artistic character portrait of a unique individual. High-quality digital art, fantasy style, detailed illustration with dramatic lighting"
        return fallback_prompt, [('style', 'artistic portrait')]

def generate_avatar_from_chat(history: list, language: str = "en", quality: str = "normal"):
    """
    Generate avatar from conversation history with AI assistant.
    
    Args:
        history: List of conversation turns [[user_msg, assistant_msg], ...]
        language: Interface language ("en" or "fr")
        quality: Generation quality ("normal" or "high")
        
    Returns:
        tuple: (generated_image, info_text)
    """
    # Extraire le prompt d'image et les éléments de la conversation
    raw_prompt, elements = extract_portrait_from_conversation(history, language)
    
    if not raw_prompt:
        return None, "Could not analyze conversation. Please continue chatting to build your portrait."
    
    # Construire un dictionnaire catégorie -> réponse à partir de la conversation
    print(f"🔍 DEBUG Raw extracted prompt: '{raw_prompt}'")
    print(f"🔍 DEBUG Elements from conversation: {elements}")
    
    # Créer le dictionnaire des réponses utilisateur
    user_responses = {}
    for category, value in elements:
        clean_category = category.lower().strip()
        clean_value = value.lower().strip()
        user_responses[clean_category] = clean_value
        print(f"🔍 DEBUG Added to dict: {clean_category} -> {clean_value}")
    
    # Construire un format "Chinese portrait" à partir du dictionnaire
    chinese_portrait_lines = []
    for category, response in user_responses.items():
        chinese_portrait_lines.append(f"If I was {category}, I would be {response}")
    
    chinese_portrait = "\n".join(chinese_portrait_lines)
    print(f"🔍 DEBUG Chinese portrait format: '{chinese_portrait}'")
    
    # Utiliser Gemma seulement pour la synthèse finale (comme le premier onglet)
    prompt = generate_flux_prompt_from_description(chinese_portrait)
    
    try:
        # Configuration selon la qualité
        if quality == "high":
            width, height, steps = 512, 512, 8
        else:
            width, height, steps = 512, 512, 4
            
        # Debug print pour voir le prompt envoyé à FLUX
        print(f"🎨 DEBUG FLUX prompt (Chat Mode): '{prompt}'")
        print(f"🎨 DEBUG FLUX prompt length: {len(prompt)} characters")
        
        # Génération avec seed aléatoire
        seed = random.randint(0, MAX_SEED)
        generator = torch.Generator(device=flux_pipe.device).manual_seed(seed)
        
        image = flux_pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator
        ).images[0]
        
        elements_text = "\n".join([f"- {category.title()}: {value}" for category, value in elements])
        
        return image, f"Avatar generated from conversation!\n\nLLM Analysis:\n{elements_text}\n\nPrompt: {prompt}\nSeed: {seed}\nQuality: {quality} ({steps} steps, {width}x{height})"
    
    except Exception as e:
        return None, f"Error during generation: {str(e)}"

def create_form_interface(language="en"):
    translations = get_translations()
    t = translations.get(language, translations["en"])
    dev_defaults = get_dev_defaults() if DEV_MODE else {}
    
    with gr.Column() as form_interface:
        gr.Markdown(f"### {t['portrait_title']}")
        
        # Commutateur de qualité
        quality_radio = gr.Radio(
            choices=["normal", "high"],
            value="normal",
            label=t["quality_label"]
        )
        
        # Groupe 1 (obligatoire)
        gr.Markdown(f"**{t['group']} 1** ⭐ *{t['required']}*")
        with gr.Row():
            if1 = gr.Textbox(label=t["if_i_was"], placeholder=t["placeholders"]["animal"], 
                           value=dev_defaults.get("if1", ""), scale=1)
            would1 = gr.Textbox(label=t["i_would_be"], placeholder=t["placeholders"]["animal_answer"], 
                              value=dev_defaults.get("would1", ""), scale=1)
        
        # Groupe 2 (obligatoire)
        gr.Markdown(f"**{t['group']} 2** ⭐ *{t['required']}*")
        with gr.Row():
            if2 = gr.Textbox(label=t["if_i_was"], placeholder=t["placeholders"]["color"], 
                           value=dev_defaults.get("if2", ""), scale=1)
            would2 = gr.Textbox(label=t["i_would_be"], placeholder=t["placeholders"]["color_answer"], 
                              value=dev_defaults.get("would2", ""), scale=1)
        
        # Groupe 3 (obligatoire)
        gr.Markdown(f"**{t['group']} 3** ⭐ *{t['required']}*")
        with gr.Row():
            if3 = gr.Textbox(label=t["if_i_was"], placeholder=t["placeholders"]["object"], 
                           value=dev_defaults.get("if3", ""), scale=1)
            would3 = gr.Textbox(label=t["i_would_be"], placeholder=t["placeholders"]["object_answer"], 
                              value=dev_defaults.get("would3", ""), scale=1)
        
        # Groupe 4 (optionnel)
        gr.Markdown(f"**{t['group']} 4** ✨ *{t['optional']}*")
        with gr.Row():
            if4 = gr.Textbox(label=t["if_i_was"], placeholder=t["placeholders"]["feeling"], 
                           value=dev_defaults.get("if4", ""), scale=1)
            would4 = gr.Textbox(label=t["i_would_be"], placeholder=t["placeholders"]["feeling_answer"], 
                              value=dev_defaults.get("would4", ""), scale=1)
        
        # Groupe 5 (optionnel)
        gr.Markdown(f"**{t['group']} 5** ✨ *{t['optional']}*")
        with gr.Row():
            if5 = gr.Textbox(label=t["if_i_was"], placeholder=t["placeholders"]["element"], 
                           value=dev_defaults.get("if5", ""), scale=1)
            would5 = gr.Textbox(label=t["i_would_be"], placeholder=t["placeholders"]["element_answer"], 
                              value=dev_defaults.get("would5", ""), scale=1)
        
        generate_btn = gr.Button(t["generate_btn"], variant="primary", size="lg")
        
        gr.Markdown(f"### {t['avatar_title']}")
        output_image = gr.Image(label=t["your_avatar"], height=400)
        output_text = gr.Textbox(label=t["information"], lines=4, interactive=False)
        
        # Hidden state for language
        lang_state = gr.State(value=language)
        
        generate_btn.click(
            fn=generate_avatar,
            inputs=[if1, would1, if2, would2, if3, would3, if4, would4, if5, would5, lang_state, quality_radio],
            outputs=[output_image, output_text]
        )
        
    return form_interface

def create_chat_interface(language="en"):
    translations = get_translations()
    t = translations.get(language, translations["en"])
    
    with gr.Column() as chat_interface:
        gr.Markdown(f"### {t['chat_title']}")
        gr.Markdown(t["chat_subtitle"])
        
        chatbot = gr.Chatbot(height=400, show_copy_button=True)
        
        # Zone de message avec bouton d'envoi
        with gr.Row():
            msg = gr.Textbox(label="Message", placeholder="Type your response here...", visible=False, scale=4)
            send_btn = gr.Button("📤", visible=False, scale=1, min_width=50)
        
        # Boutons de contrôle - en dessous du chat
        with gr.Row():
            start_btn = gr.Button("🚀 Start New Conversation", variant="primary", scale=1)
            avatar_btn = gr.Button("🎨 Get My Avatar", variant="secondary", scale=1)
            quality_chat = gr.Radio(choices=["normal", "high"], value="normal", label="Quality", scale=1)
        
        # Résultats de génération d'avatar
        avatar_output = gr.Image(label="Generated Avatar", visible=False)
        avatar_info = gr.Textbox(label="Avatar Info", lines=4, interactive=False, visible=False)
        
        # Hidden state for language
        lang_state = gr.State(value=language)
        
        def respond(message: str, history: list, language: str = "en"):
            """
            Process user message and generate simple response using get_next_category().
            
            Args:
                message: User's input message
                history: List of previous conversation turns [[user_msg, bot_msg], ...]
                language: Interface language ("en" or "fr")
                
            Yields:
                tuple: ("", updated_history) for streaming chat interface
            """
            
            # Convert history format if needed
            if history is None:
                history = []
            
            # Use simple chat logic instead of Gemma
            response = simple_chat_response(message, history)
            
            # Update history with user message and bot response
            updated_history = history + [[message, response]]
            
            # Yield the updated history (no streaming needed for simple logic)
            yield "", updated_history
        
        def start_conversation(language):
            """Démarre la conversation avec une question simple sans LLM"""
            used_categories.clear()  # Reset les catégories
            
            # Générer la première question directement
            first_category = get_next_category()
            first_question = f"If you were {first_category}, what would you be?"
            
            # Créer l'historique initial
            initial_history = [["Let's start the Chinese Portrait game!", first_question]]
            
            return initial_history, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        
        def show_avatar_interface():
            """Affiche immédiatement l'interface avatar pour montrer que ça calcule"""
            return gr.update(visible=True), gr.update(visible=True, value="Generating your avatar...")
        
        def generate_avatar_from_conversation(history, language, quality):
            if not history:
                return None, "No conversation found. Please start a conversation first."
            
            image, info = generate_avatar_from_chat(history, language, quality)
            return image, info
        
        # Événements
        start_btn.click(
            fn=start_conversation,
            inputs=[lang_state],
            outputs=[chatbot, msg, send_btn, avatar_output, avatar_info]
        )
        
        # Envoi via Enter ou bouton
        msg.submit(
            respond, 
            [msg, chatbot, lang_state], 
            [msg, chatbot], 
            queue=True
        )
        
        send_btn.click(
            respond,
            [msg, chatbot, lang_state],
            [msg, chatbot],
            queue=True
        )
        
        # Affichage immédiat de l'interface puis génération
        avatar_btn.click(
            show_avatar_interface,
            outputs=[avatar_output, avatar_info]
        ).then(
            generate_avatar_from_conversation,
            inputs=[chatbot, lang_state, quality_chat],
            outputs=[avatar_output, avatar_info]
        )
        
        gr.Markdown("*Click 'Start New Conversation' to begin, then 'Get My Avatar' when you've completed your portrait!*")
    
    return chat_interface


def detect_browser_language():
    """Détecte la langue du navigateur via JavaScript injecté"""
    # Par défaut anglais, sera override par le JavaScript
    return "en"

def create_interface(language="en"):
    translations = get_translations()
    t = translations.get(language, translations["en"])
    
    with gr.Blocks(title=t["title"], theme="gstaff/xkcd") as demo:
        gr.Markdown(f"# {t['title']}")
        gr.Markdown(t["subtitle"])
        
        with gr.Tabs():
            with gr.Tab(t["tab_form"]):
                create_form_interface(language)
            
            with gr.Tab(t["tab_chat"]):
                create_chat_interface(language)
        
        gr.Markdown("---")
        gr.Markdown(f"*{t['footer']}*")
        
        return demo


# Create the main web interface with MCP tools integrated
with gr.Blocks(title="🎭 Avatar Generator") as demo:
    gr.Markdown("# 🎭 Avatar Generator - Chinese Portrait")
    gr.Markdown("Generate personalized avatars from Chinese portrait descriptions using FLUX.1-schnell and Gemma-3n-E2B-it")
    
    with gr.Tabs():
        # Main application tabs
        with gr.Tab("📝 Form Mode"):
            create_form_interface("en")
        
        with gr.Tab("💬 Chat Mode"):
            create_chat_interface("en")
        
        
        
    
    gr.Markdown("---")
    gr.Markdown("🔌 **MCP Integration**: This app exposes tools via MCP protocol at `/gradio_api/mcp/sse`")
    gr.Markdown("*Avatar generated with FLUX.1-schnell*")

if __name__ == "__main__":
    if DEV_MODE:
        print("🚀 Running in DEV MODE with pre-filled values")
    
    print("🔌 Starting server with MCP support...")
    print("📡 MCP endpoint available at: http://localhost:7860/gradio_api/mcp/sse")
    print("🌐 Web interface available at: http://localhost:7860")
    
    demo.launch(mcp_server=True, show_api=True)