"""RAG pipeline with BioGPT for medical question answering."""

import logging
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
import re

from config import (
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    QA_MODEL_NAME,
    MAX_CONTEXT_LENGTH,
    TEMPERATURE,
    MAX_NEW_TOKENS,
    RERANK_TOP_K,
    WINDOW_SIZE_TOKENS,
    WINDOW_STRIDE,
    MAX_ANSWER_LENGTH,
    MAX_SPANS_FOR_COMPREHENSIVE
)

logger = logging.getLogger(__name__)

class MedicalRAGPipeline:
    """RAG pipeline for medical document question answering using PubMedBERT as primary model."""
    
    def __init__(self, model_name: str = PRIMARY_MODEL):
        """Initialize the RAG pipeline with direct extraction approach."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load primary model (PubMedBERT)
        logger.info(f"Loading PubMedBERT model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("PubMedBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PubMedBERT model: {e}")
            raise

        # Load extractive QA model (BioBERT SQuAD v2)
        try:
            logger.info(f"Loading QA model: {QA_MODEL_NAME}")
            self.qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
            self.qa_model.to(self.device)
            self.qa_model.eval()
            logger.info("QA model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load QA model: {e}")
            self.qa_tokenizer = None
            self.qa_model = None
    
    def create_medical_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """Create an enhanced medical-specific prompt with question-type specific instructions."""
        
        # Format retrieved documents with better structure
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:RERANK_TOP_K], 1):
            content = doc.get('document', {}).get('content', '')
            filename = doc.get('document', {}).get('filename', 'Unknown')
            section = doc.get('document', {}).get('section_title', 'Unknown Section')
            score = doc.get('combined_score', 0.0)
            
            # Clean and truncate content for better context
            content = content.strip()
            if len(content) > 500:  # Limit each chunk to 500 chars for better focus
                content = content[:500] + "..."
            
            context_parts.append(
                f"[Source {i}] {filename} - {section} (Relevance: {score:.2f})\n{content}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Detect question type for specific prompting
        question_lower = question.lower()
        is_contraindication_question = any(word in question_lower for word in [
            'contraindication', 'contraindicated', 'should not', 'avoid', 'population',
            'who should not', 'not recommended', 'not use', 'hypersensitive', 'allergic',
            'pregnancy', 'pregnant', 'children', 'pediatric', 'elderly', 'geriatric'
        ])
        is_side_effect_question = any(word in question_lower for word in ['side effect', 'adverse', 'reaction', 'symptom', 'may cause', 'can cause'])
        
        # Create question-specific instructions
        if is_contraindication_question:
            specific_instructions = """
SPECIFIC INSTRUCTIONS FOR CONTRAINDICATION QUESTIONS:
- Look for information about who should NOT take this medication
- Focus on patient populations that are contraindicated (pregnant women, children, elderly, etc.)
- Extract warnings about drug interactions or allergies
- Look for phrases like "should not", "avoid", "not recommended", "contraindicated"
- If no specific contraindications are found, state this clearly
- Provide specific patient populations or conditions that are contraindicated

EXAMPLE FORMAT:
"Based on the medical literature, [DRUG] is contraindicated in:
- [Specific population/condition]
- [Specific population/condition]
- [Additional contraindications]"
"""
        elif is_side_effect_question:
            specific_instructions = """
SPECIFIC INSTRUCTIONS FOR SIDE EFFECT QUESTIONS:
- Look for information about adverse reactions and side effects
- Focus on common and serious side effects
- Extract information about symptoms that may occur
- Look for phrases like "side effects", "adverse reactions", "may cause", "can cause"
- If no specific side effects are found, state this clearly
- Provide specific side effects with their descriptions

EXAMPLE FORMAT:
"Based on the medical literature, [DRUG] may cause the following side effects:
- [Specific side effect]
- [Specific side effect]
- [Additional side effects]"
"""
        else:
            specific_instructions = """
SPECIFIC INSTRUCTIONS:
- Extract relevant information that directly addresses the question
- Focus on factual, evidence-based information from the documents
- Use precise medical terminology when appropriate
- Structure your response logically
"""
        
        # Enhanced medical-specific prompt template with better instructions
        if is_contraindication_question:
            prompt = f"""<|endoftext|>You are a medical AI assistant. Analyze the provided medical documents to answer this specific contraindication question.

MEDICAL LITERATURE CONTEXT:
{context}

QUESTION: {question}

CRITICAL INSTRUCTIONS FOR CONTRAINDICATION QUESTIONS:
1. Look ONLY for information about who should NOT take this medication
2. Search for warnings, restrictions, or limitations on use
3. Look for phrases like "should not", "avoid", "not recommended", "contraindicated", "not use"
4. Focus on patient populations: pregnant women, children, elderly, people with certain conditions
5. If you find ANY contraindication information, extract it specifically
6. If no contraindications are found, clearly state this

ANSWER FORMAT:
"Based on the medical literature, [DRUG] is contraindicated in:
- [Specific population/condition with reason]
- [Additional contraindications]"

If no contraindications are found, say: "No specific contraindications were found in the provided documents."

ANSWER:"""
        else:
            prompt = f"""<|endoftext|>You are a specialized medical AI assistant with expertise in pharmaceutical and clinical literature analysis. Your role is to provide accurate, evidence-based answers using only the provided medical documents.

MEDICAL LITERATURE CONTEXT:
{context}

MEDICAL QUESTION: {question}

{specific_instructions}

GENERAL ANALYSIS INSTRUCTIONS:
1. Carefully analyze the provided medical literature context
2. Extract relevant information that directly addresses the question
3. Focus on factual, evidence-based information from the documents
4. Use precise medical terminology when appropriate
5. If multiple sources provide conflicting information, note this
6. If the context is insufficient, clearly state what information is missing
7. Always emphasize that patients should consult healthcare professionals for medical decisions

RESPONSE REQUIREMENTS:
- Base your answer ONLY on the provided medical literature
- Be specific and cite relevant document sources
- Use clear, professional medical language
- Structure your response logically
- If discussing contraindications, side effects, or dosing, be precise

MEDICAL ANSWER:"""
        
        return prompt
    
    def generate_answer(self, prompt: str, question: str = "") -> str:
        """Generative backup path; extractive QA is the primary path."""
        try:
            # Try PubMedBERT generation first
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=400  # Reduced to avoid tensor size issues
            )
            inputs = inputs.to(self.device)
            
            if inputs.size(1) < 10:
                logger.warning("Input prompt is too short for meaningful generation")
                return self._extract_fallback_answer(prompt, question)
            
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_beams=1
                    )
                    
                    # Decode response
                    full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer = full_response[len(prompt):].strip()
                    
                    if answer and len(answer.strip()) > 10:
                        return self.clean_answer(answer)
                    else:
                        logger.warning("PubMedBERT generated empty answer, using fallback extraction")
                        return self._extract_fallback_answer(prompt, question)
                        
                except Exception as gen_error:
                    logger.warning(f"PubMedBERT generation failed: {gen_error}")
                    return self._extract_fallback_answer(prompt, question)
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._extract_fallback_answer(prompt, question)
    
    def _extract_fallback_answer(self, prompt: str, question: str = "") -> str:
        """Extract answer directly from retrieved documents when generation fails."""
        try:
            logger.info("Using fallback extraction method")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info(f"First 200 chars of prompt: {prompt[:200]}")
            
            # Extract key information from the prompt context
            context_start = prompt.find("MEDICAL LITERATURE CONTEXT:")
            question_marker = prompt.find("QUESTION:")
            logger.info(f"Context start position: {context_start}")
            if context_start == -1:
                logger.warning("Could not find context section in prompt")
                return "I found relevant information but had difficulty generating a complete answer. Please try rephrasing your question or asking for more specific information."
            
            # Extract context between markers when available
            if question_marker != -1 and question_marker > context_start:
                context = prompt[context_start:question_marker]
            else:
                context = prompt[context_start:]
            
            # Use the passed question or extract from context
            if not question:
                question = "GRAVOL contraindications"  # Default fallback
            
            logger.info(f"Question: {question}")
            logger.info(f"Context length: {len(context)} characters")
            
            # Detect which drug the question is asking about
            question_lower = question.lower()
            target_drug = None
            if 'gravol' in question_lower:
                target_drug = 'gravol'
            elif 'xylocaine' in question_lower:
                target_drug = 'xylocaine'
            
            logger.info(f"Target drug detected: {target_drug}")
            
            # Detect question type
            is_contraindication_question = any(word in question_lower for word in [
                'contraindication', 'contraindicated', 'should not', 'avoid', 'population',
                'who should not', 'not recommended', 'not use', 'hypersensitive', 'allergic',
                'pregnancy', 'pregnant', 'children', 'pediatric', 'elderly', 'geriatric',
                'use avoid'
            ])
            
            logger.info(f"Is contraindication question: {is_contraindication_question}")
            
            # Extract contraindication information
            if is_contraindication_question:
                drug_name = target_drug.upper() if target_drug else "THE DRUG"
                contraindication_content = []
                
                # Look for contraindication information in the context
                lines = context.split('\n')
                for line in lines:
                    line_lower = line.lower()
                    contraindication_keywords = [
                        'do not', 'should not', 'avoid', 'not recommended', 'not use',
                        'contraindication', 'contraindicated', 'pregnancy', 'pregnant',
                        'children', 'pediatric', 'elderly', 'geriatric', 'glaucoma',
                        'hypersensitive', 'allergic', 'warning', 'precaution'
                    ]
                    if any(keyword in line_lower for keyword in contraindication_keywords):
                        if not target_drug or target_drug in line_lower or 'dimenhydrinate' in line_lower:
                            contraindication_content.append(line.strip())
                        else:
                            # Check surrounding context
                            context_lines = context.split('\n')
                            line_index = context_lines.index(line) if line in context_lines else -1
                            if line_index >= 0:
                                surrounding_context = ' '.join(context_lines[max(0, line_index-2):line_index+3]).lower()
                                if target_drug in surrounding_context or 'dimenhydrinate' in surrounding_context:
                                    contraindication_content.append(line.strip())
                
                if contraindication_content:
                    response = f"Based on the medical literature about {drug_name}:\n\n"
                    response += f"Contraindications and warnings for {drug_name}:\n"
                    for contra in contraindication_content[:5]:
                        cleaned_contra = self.clean_extracted_text(contra, 400)
                        response += f"- {cleaned_contra}\n"
                    response += "\nPlease consult with a healthcare professional for specific medical advice."
                    return response
            
            return "I found relevant information but had difficulty generating a complete answer. Please try rephrasing your question or asking for more specific information."
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return "I found relevant information but had difficulty generating a complete answer. Please try rephrasing your question or asking for more specific information."
    
    
    def clean_answer(self, answer: str) -> str:
        """Clean and format the generated answer."""
        # Remove any remaining prompt artifacts
        answer = re.sub(r'^.*?Answer:\s*', '', answer, flags=re.IGNORECASE)
        
        # Remove any incomplete sentences at the end
        sentences = re.split(r'[.!?]+', answer)
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            answer = '. '.join(sentences[:-1])
            if not answer.endswith('.'):
                answer += '.'
        
        # Clean up whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
    
    def clean_extracted_text(self, text: str, max_length: int = 300) -> str:
        """Clean and truncate extracted text with word boundary preservation."""
        if not text:
            return ""
        
        # Clean the text
        text = text.strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common chopping issues
        text = re.sub(r'\bular\b', 'regular', text)  # Fix "ular" -> "regular"
        text = re.sub(r'\bntreal\b', 'Montreal', text)  # Fix "ntreal" -> "Montreal"
        text = re.sub(r'\bction\b', 'action', text)  # Fix "ction" -> "action"
        text = re.sub(r'\bffect\b', 'effect', text)  # Fix "ffect" -> "effect"
        text = re.sub(r'\bm experience\b', 'experience', text)  # Fix "m experience" -> "experience"
        text = re.sub(r'\bm\s+', '', text)  # Remove standalone "m" followed by space
        
        # Clean up PDF bullet formatting artifacts
        text = re.sub(r'\bo\s+', '• ', text)  # Replace "o " with proper bullet "• "
        text = re.sub(r'\s+o\s+', ' • ', text)  # Replace " o " with " • "
        text = re.sub(r'^\s*o\s+', '• ', text)  # Replace leading "o " with "• "
        
        # Fix clipped fragments at the beginning
        text = re.sub(r'^\s*me\s+', 'Some ', text)  # Fix "me " -> "Some "
        text = re.sub(r'^\s*cist\s+', 'Consult ', text)  # Fix "cist " -> "Consult "
        text = re.sub(r'^\s*ud\s+', 'Including ', text)  # Fix "ud " -> "Including "
        
        # Remove other common clipped fragments at start (but not if already fixed)
        if not text.startswith('Some ') and not text.startswith('Consult ') and not text.startswith('Including '):
            text = re.sub(r'^\s*[a-z]{1,3}\s+', '', text)  # Remove 1-3 lowercase letters at start
        
        # Fix missing words in common phrases (only if not already present)
        if 'Some drugs used to treat depression' not in text:
            text = re.sub(r'\bdrugs used to treat depression\b', 'Some drugs used to treat depression', text)
        if 'Drugs used to help you sleep' not in text:
            text = re.sub(r'\bdrugs used to help you sleep\b', 'Drugs used to help you sleep', text)
        if 'Drugs used to reduce tension' not in text:
            text = re.sub(r'\bdrugs used to reduce tension\b', 'Drugs used to reduce tension', text)
        
        # If text is too long, truncate at word boundary
        if len(text) > max_length:
            # Find the last complete word within the limit
            truncated = text[:max_length]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # Only truncate if we can preserve most of the text
                text = truncated[:last_space] + "..."
            else:
                text = truncated + "..."
        
        return text
    
    def _answer_with_extractive_qa(self, question: str, reranked_docs: List[Dict]) -> Optional[str]:
        """Answer using extractive QA over top reranked documents with sliding window."""
        if not getattr(self, 'qa_model', None) or not getattr(self, 'qa_tokenizer', None):
            return None
        contexts: List[Tuple[str, Dict]] = []
        for doc in reranked_docs[:RERANK_TOP_K]:
            content = doc['document'].get('content', '')
            if content:
                contexts.append((content, doc['document']))

        if not contexts:
            return None

        # Detect question type for comprehensive answers
        q_lower = question.lower()
        
        # Contraindication questions
        prefer_contra = any(t in q_lower for t in [
            'contraindication', 'contraindicated', 'should not', 'avoid', 'not recommended',
            'not use', 'warning', 'precaution', 'pregnancy', 'children', 'elderly', 'allergic'
        ])
        
        # Mechanism of action questions
        prefer_moa = any(t in q_lower for t in [
            'how does', 'mechanism', 'work', 'works', 'mode of action', 'moa'
        ]) and 'contraindicat' not in q_lower
        
        # Clinical use/indication questions
        prefer_indication = any(t in q_lower for t in [
            'clinical use', 'used for', 'indication', 'indications', 'what is used', 
            'purpose', 'treat', 'treatment', 'prevent', 'relieve'
        ]) and 'how' not in q_lower and 'mechanism' not in q_lower and 'work' not in q_lower
        
        # Dosage questions
        prefer_dosage = any(t in q_lower for t in [
            'dose', 'dosage', 'how much', 'administration', 'administer', 'take',
            'mg', 'ml', 'tablet', 'capsule', 'injection', 'frequency'
        ])
        
        # Side effect questions
        prefer_side_effects = any(t in q_lower for t in [
            'side effect', 'adverse', 'reaction', 'symptom', 'may cause', 'can cause',
            'complication', 'toxicity', 'harmful'
        ])
        
        # Drug interaction questions
        prefer_interactions = any(t in q_lower for t in [
            'interaction', 'drug interaction', 'interact', 'combine', 'mix',
            'with other drugs', 'concurrent', 'simultaneous'
        ])
        
        # Storage questions
        prefer_storage = any(t in q_lower for t in [
            'storage', 'store', 'temperature', 'refrigerate', 'room temperature',
            'expiry', 'expiration', 'shelf life'
        ])
        
        # General information questions
        prefer_general = any(t in q_lower for t in [
            'what is', 'tell me about', 'information about', 'describe', 'explain',
            'overview', 'summary', 'details'
        ]) and not any([prefer_contra, prefer_moa, prefer_indication, prefer_dosage, 
                       prefer_side_effects, prefer_interactions, prefer_storage])
        
        # Check if user wants comprehensive information
        is_comprehensive = any(t in q_lower for t in [
            'all information', 'all details', 'everything', 'complete', 'comprehensive',
            'give me all', 'tell me all', 'all about', 'all warnings', 'all precautions'
        ])
        
        # Debug logging to see which question type is detected
        logger.info(f"Question type detection - Contra: {prefer_contra}, MOA: {prefer_moa}, Indication: {prefer_indication}, "
                   f"Dosage: {prefer_dosage}, Side effects: {prefer_side_effects}, Interactions: {prefer_interactions}, "
                   f"Storage: {prefer_storage}, General: {prefer_general}")
        
        if is_comprehensive:
            return self._extract_multiple_spans(question, contexts, prefer_contra, prefer_moa, prefer_indication, 
                                              prefer_dosage, prefer_side_effects, prefer_interactions, prefer_storage, prefer_general)
        else:
            return self._extract_single_best_span(question, contexts, prefer_contra, prefer_moa, prefer_indication,
                                                prefer_dosage, prefer_side_effects, prefer_interactions, prefer_storage, prefer_general)
    
    def _extract_single_best_span(self, question: str, contexts: List[Tuple[str, Dict]], 
                                 prefer_contra: bool, prefer_moa: bool, prefer_indication: bool = False,
                                 prefer_dosage: bool = False, prefer_side_effects: bool = False,
                                 prefer_interactions: bool = False, prefer_storage: bool = False,
                                 prefer_general: bool = False) -> Optional[str]:
        """Extract the single best span for focused questions."""
        best_answer = ""
        best_score = float('-inf')

        for content, _meta in contexts:
            inputs = self.qa_tokenizer(
                [question],
                [content],
                return_tensors="pt",
                truncation=True,
                max_length=WINDOW_SIZE_TOKENS,
                stride=WINDOW_STRIDE,
                padding='max_length',
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=True
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            offset_mapping = inputs['offset_mapping']
            token_type_ids = inputs.get('token_type_ids')

            with torch.no_grad():
                outputs = self.qa_model(input_ids=input_ids, attention_mask=attention_mask)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

            for i in range(input_ids.size(0)):
                start_logit = start_logits[i].clone()
                end_logit = end_logits[i].clone()
                max_len = min(MAX_ANSWER_LENGTH, input_ids.size(1))
                offsets = offset_mapping[i]

                # Mask out non-context tokens (token_type_ids != 1) and padding
                if token_type_ids is not None:
                    tti = token_type_ids[i]
                    context_mask = (tti == 1) & (attention_mask[i] == 1)
                else:
                    # Fallback: use offsets to drop (0,0) and keep attention_mask
                    context_mask = (attention_mask[i] == 1)
                    # zero offsets usually mean special/pad
                    zero_offsets = torch.tensor([o[0] == 0 and o[1] == 0 for o in offsets], device=context_mask.device)
                    context_mask = context_mask & (~zero_offsets)

                invalid = ~context_mask
                start_logit[invalid] = float('-inf')
                end_logit[invalid] = float('-inf')

                # Try top-k start positions to find a good span
                k = min(5, start_logit.numel())
                topk_vals, topk_idx = torch.topk(start_logit, k)
                chosen_span = None
                chosen_score = float('-inf')

                for j in range(k):
                    s_idx = topk_idx[j].item()
                    e_candidates = end_logit[s_idx:s_idx + max_len]
                    if e_candidates.numel() == 0:
                        continue
                    e_rel = torch.argmax(e_candidates).item()
                    e_idx = s_idx + e_rel
                    if e_idx < s_idx or e_idx >= len(offsets):
                        continue
                    s_char = offsets[s_idx][0]
                    e_char = offsets[e_idx][1]
                    span = content[s_char:e_char].strip()
                    if not span or len(span) <= 3:
                        continue
                    score = start_logit[s_idx].item() + end_logit[e_idx].item()
                    # Apply preference scoring based on question type
                    span_l = span.lower()
                    
                    if prefer_contra:
                        if any(k in span_l for k in ['contraindicat', 'should not', 'avoid', 'not recommended', 'not use', 'warning', 'precaution']):
                            score += 2.0
                        else:
                            if any(k in span_l for k in ['indication', 'how it works', 'works by']):
                                score -= 1.5
                    elif prefer_moa:
                        if any(k in span_l for k in ['works by', 'mechanism', 'blocks', 'antagonist', 'antihistamine', 'h1', 'receptor', 'vestibular', 'inner ear', 'vomiting reflex']):
                            score += 1.5
                        if any(k in span_l for k in ['used to', 'prevent and relieve symptoms', 'used for']):
                            score -= 1.0
                    elif prefer_indication:
                        if any(k in span_l for k in ['used to', 'prevent and relieve', 'treat', 'treatment', 'indication', 'symptoms such as', 'nausea', 'vomiting', 'vertigo', 'motion sickness']):
                            score += 2.0
                        if any(k in span_l for k in ['works by', 'mechanism', 'blocks', 'affecting the brain']):
                            score -= 1.5
                    elif prefer_dosage:
                        if any(k in span_l for k in ['dose', 'dosage', 'mg', 'ml', 'tablet', 'capsule', 'injection', 'frequency', 'daily', 'twice', 'three times']):
                            score += 2.0
                        if any(k in span_l for k in ['contraindication', 'side effect', 'mechanism']):
                            score -= 1.0
                    elif prefer_side_effects:
                        if any(k in span_l for k in ['side effect', 'adverse', 'reaction', 'symptom', 'may cause', 'can cause', 'complication', 'toxicity']):
                            score += 2.0
                        if any(k in span_l for k in ['indication', 'mechanism', 'dosage']):
                            score -= 1.0
                    elif prefer_interactions:
                        if any(k in span_l for k in ['interaction', 'drug interaction', 'interact', 'combine', 'mix', 'concurrent', 'simultaneous']):
                            score += 2.0
                        if any(k in span_l for k in ['indication', 'mechanism', 'dosage']):
                            score -= 1.0
                    elif prefer_storage:
                        if any(k in span_l for k in ['storage', 'store', 'temperature', 'refrigerate', 'room temperature', 'expiry', 'expiration']):
                            score += 2.0
                        if any(k in span_l for k in ['indication', 'mechanism', 'dosage', 'side effect']):
                            score -= 1.0
                    elif prefer_general:
                        # For general questions, prefer comprehensive information
                        if any(k in span_l for k in ['used to', 'indication', 'mechanism', 'dosage', 'side effect', 'contraindication']):
                            score += 1.0
                    else:
                        # Fallback: no specific bias, rely on QA model's natural scoring
                        pass
                    if score > chosen_score:
                        chosen_score = score
                        chosen_span = span

                if chosen_span and chosen_score > best_score:
                    # Expand to sentence boundaries for readability
                    start_idx = content.rfind('.', 0, content.find(chosen_span))
                    end_idx = content.find('.', content.find(chosen_span) + len(chosen_span))
                    start_idx = 0 if start_idx == -1 else start_idx + 1
                    end_idx = len(content) if end_idx == -1 else end_idx + 1
                    sentence_span = content[start_idx:end_idx].strip()
                    # Trim at section/header cues to avoid trailing unrelated text
                    stop_cues = [
                        'What are', 'What is', 'Ingredients', 'Source Documents:', '[Source', '\n1.', '\n2.', '\n- '
                    ]
                    cut_pos = len(sentence_span)
                    for cue in stop_cues:
                        pos = sentence_span.find(cue)
                        if pos != -1:
                            cut_pos = min(cut_pos, pos)
                    sentence_span = sentence_span[:cut_pos].strip()
                    # Limit to first 1-2 relevant sentences
                    parts = re.split(r'(?<=[.!?])\s+', sentence_span)
                    if prefer_contra or prefer_moa:
                        sentence_span = ' '.join(parts[:2]).strip()
                    else:
                        sentence_span = parts[0].strip()
                    best_score = chosen_score
                    best_answer = sentence_span if len(sentence_span) >= 3 else chosen_span

        if best_answer:
            return self.clean_extracted_text(best_answer, 800)
        return None
    
    def _extract_multiple_spans(self, question: str, contexts: List[Tuple[str, Dict]], 
                              prefer_contra: bool, prefer_moa: bool, prefer_indication: bool = False,
                              prefer_dosage: bool = False, prefer_side_effects: bool = False,
                              prefer_interactions: bool = False, prefer_storage: bool = False,
                              prefer_general: bool = False) -> Optional[str]:
        """Extract multiple relevant spans for comprehensive questions."""
        all_spans = []
        
        for content, _meta in contexts:
            inputs = self.qa_tokenizer(
                [question],
                [content],
                return_tensors="pt",
                truncation=True,
                max_length=WINDOW_SIZE_TOKENS,
                stride=WINDOW_STRIDE,
                padding='max_length',
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=True
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            offset_mapping = inputs['offset_mapping']
            token_type_ids = inputs.get('token_type_ids')

            with torch.no_grad():
                outputs = self.qa_model(input_ids=input_ids, attention_mask=attention_mask)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

            for i in range(input_ids.size(0)):
                start_logit = start_logits[i].clone()
                end_logit = end_logits[i].clone()
                max_len = min(MAX_ANSWER_LENGTH, input_ids.size(1))
                offsets = offset_mapping[i]

                # Mask out non-context tokens
                if token_type_ids is not None:
                    tti = token_type_ids[i]
                    context_mask = (tti == 1) & (attention_mask[i] == 1)
                else:
                    context_mask = (attention_mask[i] == 1)
                    zero_offsets = torch.tensor([o[0] == 0 and o[1] == 0 for o in offsets], device=context_mask.device)
                    context_mask = context_mask & (~zero_offsets)

                invalid = ~context_mask
                start_logit[invalid] = float('-inf')
                end_logit[invalid] = float('-inf')

                # Extract multiple good spans
                k = min(10, start_logit.numel())
                topk_vals, topk_idx = torch.topk(start_logit, k)
                
                for j in range(k):
                    s_idx = topk_idx[j].item()
                    e_candidates = end_logit[s_idx:s_idx + max_len]
                    if e_candidates.numel() == 0:
                        continue
                    e_rel = torch.argmax(e_candidates).item()
                    e_idx = s_idx + e_rel
                    if e_idx < s_idx or e_idx >= len(offsets):
                        continue
                    s_char = offsets[s_idx][0]
                    e_char = offsets[e_idx][1]
                    span = content[s_char:e_char].strip()
                    if not span or len(span) <= 5:
                        continue
                    
                    score = start_logit[s_idx].item() + end_logit[e_idx].item()
                    
                    # Apply preference scoring based on question type
                    span_l = span.lower()
                    
                    if prefer_contra:
                        if any(k in span_l for k in ['contraindicat', 'should not', 'avoid', 'not recommended', 'not use', 'warning', 'precaution']):
                            score += 2.0
                    elif prefer_moa:
                        if any(k in span_l for k in ['works by', 'mechanism', 'blocks', 'antagonist', 'antihistamine', 'h1', 'receptor', 'vestibular', 'inner ear', 'vomiting reflex']):
                            score += 1.5
                    elif prefer_indication:
                        if any(k in span_l for k in ['used to', 'prevent and relieve', 'treat', 'treatment', 'indication', 'symptoms such as', 'nausea', 'vomiting', 'vertigo', 'motion sickness']):
                            score += 2.0
                    elif prefer_dosage:
                        if any(k in span_l for k in ['dose', 'dosage', 'mg', 'ml', 'tablet', 'capsule', 'injection', 'frequency', 'daily', 'twice', 'three times']):
                            score += 2.0
                    elif prefer_side_effects:
                        if any(k in span_l for k in ['side effect', 'adverse', 'reaction', 'symptom', 'may cause', 'can cause', 'complication', 'toxicity']):
                            score += 2.0
                    elif prefer_interactions:
                        if any(k in span_l for k in ['interaction', 'drug interaction', 'interact', 'combine', 'mix', 'concurrent', 'simultaneous']):
                            score += 2.0
                    elif prefer_storage:
                        if any(k in span_l for k in ['storage', 'store', 'temperature', 'refrigerate', 'room temperature', 'expiry', 'expiration']):
                            score += 2.0
                    elif prefer_general:
                        # For general questions, prefer comprehensive information
                        if any(k in span_l for k in ['used to', 'indication', 'mechanism', 'dosage', 'side effect', 'contraindication']):
                            score += 1.0
                    else:
                        # Fallback: no specific bias, rely on QA model's natural scoring
                        pass
                    
                    # Expand to full sentences
                    start_idx_text = content.rfind('.', 0, s_char)
                    end_idx_text = content.find('.', e_char)
                    start_idx_text = 0 if start_idx_text == -1 else start_idx_text + 1
                    end_idx_text = len(content) if end_idx_text == -1 else end_idx_text + 1
                    full_sentence = content[start_idx_text:end_idx_text].strip()
                    
                    # Clean the sentence
                    stop_cues = ['What are', 'What is', 'Ingredients', 'Source Documents:', '[Source']
                    for cue in stop_cues:
                        pos = full_sentence.find(cue)
                        if pos != -1:
                            full_sentence = full_sentence[:pos].strip()
                    
                    if len(full_sentence) > 10:
                        all_spans.append((full_sentence, score))
        
        # Sort by score and remove duplicates
        all_spans.sort(key=lambda x: x[1], reverse=True)
        unique_spans = []
        seen_content = set()
        
        for span, score in all_spans:
            # Clean the span first
            cleaned_span = self.clean_extracted_text(span, 500)
            span_key = cleaned_span.lower().strip()
            
            # More aggressive deduplication - check for substantial overlap
            is_duplicate = False
            for seen in seen_content:
                if len(span_key) > 20 and len(seen) > 20:
                    # Check if spans are substantially similar (80% overlap)
                    words1 = set(span_key.split())
                    words2 = set(seen.split())
                    if len(words1) > 5 and len(words2) > 5:
                        overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
                        if overlap > 0.8:
                            is_duplicate = True
                            break
            
            if not is_duplicate and len(span_key) > 10:
                unique_spans.append(cleaned_span)
                seen_content.add(span_key)
                if len(unique_spans) >= MAX_SPANS_FOR_COMPREHENSIVE:
                    break
        
        if unique_spans:
            # Format as bullet points
            formatted_answer = "\n".join(f"• {span}" for span in unique_spans)
            return self.clean_extracted_text(formatted_answer, 2000)
        
        return None

    def rerank_documents(self, semantic_results: List[Tuple[float, Dict]], 
                        keyword_results: List[Dict], question: str = "") -> List[Dict]:
        """Enhanced reranking based on combined semantic and keyword relevance with drug-specific boosting."""
        
        # Create a combined score mapping
        combined_scores = {}
        
        # Normalize semantic scores to 0-1 range
        if semantic_results:
            max_semantic = max(score for score, _ in semantic_results)
            min_semantic = min(score for score, _ in semantic_results)
            semantic_range = max_semantic - min_semantic if max_semantic != min_semantic else 1
        else:
            semantic_range = 1
        
        # Add semantic scores with normalization
        for score, metadata in semantic_results:
            chunk_id = metadata.get('chunk_id')
            if chunk_id:
                normalized_score = (score - min_semantic) / semantic_range if semantic_range > 0 else 0
                combined_scores[chunk_id] = {
                    'semantic_score': normalized_score,
                    'keyword_score': 0,
                    'metadata': metadata,
                    'original_semantic': score
                }
        
        # Normalize keyword scores to 0-1 range
        if keyword_results:
            keyword_scores = [result.get('score', 0) for result in keyword_results]
            max_keyword = max(keyword_scores)
            min_keyword = min(keyword_scores)
            keyword_range = max_keyword - min_keyword if max_keyword != min_keyword else 1
        else:
            keyword_range = 1
        
        # Add keyword scores with normalization
        for result in keyword_results:
            chunk_id = result.get('document', {}).get('chunk_id')
            if chunk_id:
                original_score = result.get('score', 0)
                normalized_score = (original_score - min_keyword) / keyword_range if keyword_range > 0 else 0
                
                if chunk_id in combined_scores:
                    combined_scores[chunk_id]['keyword_score'] = normalized_score
                else:
                    combined_scores[chunk_id] = {
                        'semantic_score': 0,
                        'keyword_score': normalized_score,
                        'metadata': result.get('document', {}),
                        'original_keyword': original_score
                    }
        
        # Extract drug names from question for boosting
        question_lower = question.lower()
        target_drugs = []
        if 'gravol' in question_lower:
            target_drugs.append('gravol')
        if 'xylocaine' in question_lower:
            target_drugs.append('xylocaine')
        if 'lidocaine' in question_lower:
            target_drugs.append('lidocaine')
        if 'tapazole' in question_lower:
            target_drugs.append('tapazole')
        
        # Detect question type for enhanced boosting
        is_contraindication_question = any(word in question_lower for word in [
            'contraindication', 'contraindicated', 'should not', 'avoid', 'population',
            'who should not', 'not recommended', 'not use', 'hypersensitive', 'allergic',
            'pregnancy', 'pregnant', 'children', 'pediatric', 'elderly', 'geriatric'
        ])
        is_side_effect_question = any(word in question_lower for word in ['side effect', 'adverse', 'reaction', 'symptom', 'may cause', 'can cause'])
        
        # Calculate enhanced combined scores
        reranked = []
        for chunk_id, scores in combined_scores.items():
            # Enhanced weighting: 60% semantic, 40% keyword for better balance
            combined_score = (scores['semantic_score'] * 0.6 + 
                            scores['keyword_score'] * 0.4)
            
            content = scores['metadata'].get('content', '').lower()
            
            # Boost score for medical terminology matches
            medical_terms = ['contraindication', 'dosage', 'side effect', 'indication', 
                           'administration', 'precaution', 'warning', 'interaction']
            medical_boost = sum(1 for term in medical_terms if term in content) * 0.05
            combined_score += min(medical_boost, 0.2)  # Cap boost at 0.2
            
            # MASSIVE BOOST: Drug-specific content matching with question-type awareness
            drug_boost = 0.0
            for drug in target_drugs:
                if drug in content:
                    drug_boost += 0.5  # Major boost for exact drug match
                    
                    # MASSIVE BOOST for contraindication questions
                    if is_contraindication_question:
                        # Look for contraindication-specific content
                        contraindication_keywords = [
                            'contraindication', 'contraindicated', 'should not', 'avoid', 
                            'not recommended', 'not use', 'hypersensitive', 'allergic',
                            'pregnancy', 'pregnant', 'children', 'pediatric', 'elderly', 
                            'geriatric', 'breastfeeding', 'lactation', 'warnings', 'precautions'
                        ]
                        
                        contraindication_matches = sum(1 for keyword in contraindication_keywords if keyword in content)
                        if contraindication_matches > 0:
                            drug_boost += 1.0  # MASSIVE boost for contraindication content
                            drug_boost += contraindication_matches * 0.2  # Additional boost per match
                        
                        # Extra boost for section titles containing contraindication
                        section_title = scores['metadata'].get('section_title', '').lower()
                        if any(keyword in section_title for keyword in ['contraindication', 'warning', 'precaution']):
                            drug_boost += 0.5  # Extra boost for relevant section titles
                    
                    # Boost for side effect questions
                    elif is_side_effect_question:
                        side_effect_keywords = ['side effect', 'adverse', 'reaction', 'symptom', 'may cause', 'can cause']
                        side_effect_matches = sum(1 for keyword in side_effect_keywords if keyword in content)
                        if side_effect_matches > 0:
                            drug_boost += 0.8  # Major boost for side effect content
                            drug_boost += side_effect_matches * 0.15  # Additional boost per match
            
            # Force-boost chunks whose section title clearly indicates contraindications or mechanism-of-action
            section_title = scores['metadata'].get('section_title', '').lower()
            if any(kw in section_title for kw in ['contraindication', 'contra-indication', 'contra indications']):
                combined_score += 0.8
            if any(kw in section_title for kw in ['how does', 'mechanism', 'mode of action', 'moa']):
                combined_score += 0.4

            combined_score += min(drug_boost, 2.0)  # Increased cap for drug boost
            
            reranked.append({
                'chunk_id': chunk_id,
                'combined_score': combined_score,
                'semantic_score': scores['semantic_score'],
                'keyword_score': scores['keyword_score'],
                'document': scores['metadata']
            })
        
        # Sort by combined score
        reranked.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top results, ensuring we have at least some results
        return reranked[:max(RERANK_TOP_K, 3)]
    
    def answer_question(self, question: str, 
                       semantic_results: List[Tuple[float, Dict]],
                       keyword_results: List[Dict]) -> Dict:
        """Answer a medical question using retrieved documents with enhanced error handling."""
        
        logger.info(f"Answering question: {question}")
        
        try:
            # Validate inputs
            if not question or not question.strip():
                return {
                    'answer': 'Please provide a valid question.',
                    'sources': [],
                    'confidence': 0.0,
                    'num_sources': 0
                }
            
            # Rerank documents with question context for drug-specific boosting
            reranked_docs = self.rerank_documents(semantic_results, keyword_results, question)
            
            if not reranked_docs:
                return {
                    'answer': 'I could not find relevant information to answer your question. Please try rephrasing your question or check if your documents contain information about this topic.',
                    'sources': [],
                    'confidence': 0.0,
                    'num_sources': 0
                }
            
            # Check if we have enough context
            total_context_length = sum(len(doc['document'].get('content', '')) for doc in reranked_docs)
            if total_context_length < 50:
                return {
                    'answer': 'I found some relevant documents but they contain insufficient information to provide a comprehensive answer. Please try asking a more specific question.',
                    'sources': [],
                    'confidence': 0.1,
                    'num_sources': len(reranked_docs)
                }
            
            # If contraindication-style question, filter docs to relevant sections first
            ql = question.lower()
            is_contra = any(w in ql for w in [
                'contraindication', 'contraindicated', 'should not', 'avoid', 'not recommended',
                'not use', 'warning', 'precaution'
            ])
            narrowed_docs: List[Dict] = []
            if is_contra:
                for d in reranked_docs:
                    section = d['document'].get('section_title', '')
                    content = d['document'].get('content', '')
                    s = section.lower()
                    c = content.lower()
                    if any(k in s for k in ['contraindication', 'contra-indication', 'warnings', 'precautions']) or \
                       any(k in c for k in ['contraindicat', 'should not', 'avoid', 'not recommended', 'not use', 'warning', 'precaution']):
                        narrowed_docs.append(d)
            primary_docs = narrowed_docs if narrowed_docs else reranked_docs

            # Try extractive QA first
            answer = self._answer_with_extractive_qa(question, primary_docs)
            
            # If extractive QA fails, fall back to generative prompt then rule-based
            if not answer:
                prompt = self.create_medical_prompt(question, reranked_docs)
                if prompt and len(prompt.strip()) >= 50:
                    answer = self.generate_answer(prompt, question)
            
            # Validate answer
            if not answer or len(answer.strip()) < 10:
                return {
                    'answer': 'I found relevant information but had difficulty generating a complete answer. Please try rephrasing your question or asking for more specific information.',
                    'sources': [],
                    'confidence': 0.2,
                    'num_sources': len(reranked_docs)
                }
            
            # Prepare sources with better formatting
            sources = []
            for i, doc in enumerate(reranked_docs):
                content = doc['document'].get('content', '')
                sources.append({
                    'filename': doc['document'].get('filename', 'Unknown'),
                    'section': doc['document'].get('section_title', 'Unknown'),
                    'page': doc['document'].get('page_number', 'Unknown'),
                    'score': round(doc['combined_score'], 3),
                    'content_preview': content[:300] + '...' if len(content) > 300 else content
                })
            
            # Calculate confidence based on multiple factors
            top_score = reranked_docs[0]['combined_score'] if reranked_docs else 0.0
            num_sources = len(reranked_docs)
            answer_length = len(answer.strip())
            
            # Enhanced confidence calculation
            confidence = min(top_score, 1.0)
            if num_sources >= 3:
                confidence += 0.1  # Boost for multiple sources
            if answer_length > 100:
                confidence += 0.05  # Boost for detailed answers
            confidence = min(confidence, 1.0)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': round(confidence, 2),
                'num_sources': num_sources
            }
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                'answer': f'I encountered an error while processing your question: {str(e)}. Please try again with a different question.',
                'sources': [],
                'confidence': 0.0,
                'num_sources': 0
            }
    
    def batch_answer_questions(self, questions: List[str],
                              semantic_results_list: List[List[Tuple[float, Dict]]],
                              keyword_results_list: List[List[Dict]]) -> List[Dict]:
        """Answer multiple questions in batch."""
        
        results = []
        for question, semantic_results, keyword_results in zip(
            questions, semantic_results_list, keyword_results_list
        ):
            result = self.answer_question(question, semantic_results, keyword_results)
            results.append(result)
        
        return results
