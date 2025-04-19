import re
import json
#import yaml
import os
import pandas as pd
import streamlit as st
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#import google.generativeai as genai

# Download NLTK resources
nltk.download('punkt', quiet=True)


# =====================
# Pattern Matching Approach
# =====================

def load_profanity_list():
    """Load a list of profane words"""
    profane_words = [
        "fuck", "shit", "damn", "bitch", "asshole", "crap", "hell",
        "bastard", "dick", "piss", "cunt", "bullshit", "dumbass",
        "motherfucker", "goddamn", "ass", "whore", "slut"
    ]

    variations = []
    for word in profane_words:
        variations.append(word)
        for i in range(len(word)):
            if i > 0 and i < len(word) - 1:
                variations.append(word[:i] + '*' + word[i + 1:])
        variations.append(word.replace('a', '4').replace('e', '3').replace('i', '1').replace('o', '0'))

    return list(set(variations))


def detect_profanity_regex(text, profanity_list):
    """Detect profane language using regex pattern matching"""
    if not text or not isinstance(text, str):
        return False, []

    text = text.lower()
    found_words = []

    for word in profanity_list:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text):
            found_words.append(word)

    return len(found_words) > 0, found_words


def analyze_profanity_pattern_matching(conversations):
    """Analyze all conversations for profanity using pattern matching"""
    profanity_list = load_profanity_list()
    results = {
        'agent_profanity': defaultdict(list),
        'borrower_profanity': defaultdict(list)
    }

    for call_id, conversation in conversations.items():
        for utterance in conversation:
            speaker = utterance.get('speaker', '').lower()
            text = utterance.get('text', '')

            has_profanity, found_words = detect_profanity_regex(text, profanity_list)

            if has_profanity:
                if 'agent' in speaker:
                    results['agent_profanity'][call_id].append({
                        'text': text,
                        'profane_words': found_words,
                        'stime': utterance.get('stime'),
                        'etime': utterance.get('etime')
                    })
                elif any(term in speaker.lower() for term in ['customer', 'borrower']):
                    results['borrower_profanity'][call_id].append({
                        'text': text,
                        'profane_words': found_words,
                        'stime': utterance.get('stime'),
                        'etime': utterance.get('etime')
                    })

    return results


# =====================
# Machine Learning Approach
# =====================

class MLProfanityDetector:
    """A simple ML-based profanity detector using word embeddings"""

    def __init__(self):
        """Initialize the ML profanity detector"""
        self.profanity_list = load_profanity_list()
        self.contextual_indicators = [
            "angry", "upset", "hate", "stupid", "idiot", "dumb",
            "annoying", "useless", "terrible", "horrible", "worst"
        ]

    def predict(self, text):
        """Predict if text contains profanity"""
        if not text or not isinstance(text, str):
            return False, []

        text = text.lower()
        tokens = word_tokenize(text)

        found_words = []
        for word in self.profanity_list:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text):
                found_words.append(word)

        contextual_score = sum(1 for word in tokens if word in self.contextual_indicators)

        if found_words or contextual_score >= 2:
            return True, found_words if found_words else ["contextual offensive language"]
        else:
            return False, []


def analyze_profanity_ml(conversations):
    """Analyze all conversations for profanity using the ML approach"""
    detector = MLProfanityDetector()
    results = {
        'agent_profanity': defaultdict(list),
        'borrower_profanity': defaultdict(list)
    }

    for call_id, conversation in conversations.items():
        for utterance in conversation:
            speaker = utterance.get('speaker', '').lower()
            text = utterance.get('text', '')

            has_profanity, found_words = detector.predict(text)

            if has_profanity:
                if 'agent' in speaker:
                    results['agent_profanity'][call_id].append({
                        'text': text,
                        'profane_words': found_words,
                        'stime': utterance.get('stime'),
                        'etime': utterance.get('etime')
                    })
                elif any(term in speaker.lower() for term in ['customer', 'borrower']):
                    results['borrower_profanity'][call_id].append({
                        'text': text,
                        'profane_words': found_words,
                        'stime': utterance.get('stime'),
                        'etime': utterance.get('etime')
                    })

    return results


# =====================
# Privacy and Compliance Violation
# =====================

def detect_privacy_violation(text):
    """Detect privacy violation in the text (e.g., sharing account details without verification)"""
    sensitive_keywords = ['balance', 'account number', 'social security', 'ssn', 'date of birth', 'address',
                          'payment information']
    verification_keywords = ['date of birth', 'address', 'social security number', 'ssn']

    # Check for sensitive information being mentioned without verification
    sensitive_found = any(keyword in text.lower() for keyword in sensitive_keywords)
    verification_found = any(verification_keyword in text.lower() for verification_keyword in verification_keywords)

    if sensitive_found and not verification_found:
        return True
    return False


def analyze_privacy_violation(conversations):
    """Analyze all conversations for privacy violations"""
    results = {
        'privacy_violation': defaultdict(list)
    }

    for call_id, conversation in conversations.items():
        for utterance in conversation:
            speaker = utterance.get('speaker', '').lower()
            text = utterance.get('text', '')

            if detect_privacy_violation(text):
                results['privacy_violation'][call_id].append({
                    'text': text,
                    'stime': utterance.get('stime'),
                    'etime': utterance.get('etime')
                })

    return results


# =====================
# Data Loading Functions
# =====================

def load_data(file_content):
    """Load conversation data from a JSON string or file"""
    try:
        if isinstance(file_content, str):
            return json.loads(file_content)
        else:
            data = json.load(file_content)
            return data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# =====================
# Streamlit App
# =====================

def run_conversation_analysis_app():
    st.title("Debt Collection Call Analysis")
    st.subheader("Conversation Analysis")

    uploaded_file = st.file_uploader("Upload a conversation file (JSON)", type=["json"])

    # Select analysis type
    entity_type = st.selectbox(
        "Select Analysis Type",
        ["Profanity Detection", "Privacy and Compliance Violation"]
    )

    # Conditionally display the "Analysis Approach" dropdown based on the "Analysis Type"
    if entity_type == "Profanity Detection":
        approach = st.selectbox(
            "Select Analysis Approach",
            ["Pattern Matching", "Machine Learning", "LLM-based"]
        )

    if uploaded_file is not None:
        conversation_data = load_data(uploaded_file)

        if conversation_data:
            call_id = uploaded_file.name.split('.')[0]
            conversations = {call_id: conversation_data}

            if st.button("Analyze"):
                st.subheader("Analysis Results")

                if entity_type == "Profanity Detection":
                    if approach == "Pattern Matching":
                        results = analyze_profanity_pattern_matching(conversations)
                    elif approach == "Machine Learning":
                        results = analyze_profanity_ml(conversations)
                    else:
                        results = analyze_profanity_llm(conversations, "your-api-key")

                    agent_profanity = results['agent_profanity']
                    borrower_profanity = results['borrower_profanity']

                    if not agent_profanity and not borrower_profanity:
                        st.success("No profanity detected in the conversation.")
                    else:
                        if agent_profanity:
                            st.error("Agent Profanity Detected:")
                            for call_id, instances in agent_profanity.items():
                                st.write(f"**Call ID: {call_id}**")
                                for i, instance in enumerate(instances):
                                    st.write(f"Instance {i + 1}:")
                                    st.write(f"- Text: *{instance['text']}*")
                                    st.write(f"- Profane words: {', '.join(instance['profane_words'])}")
                                    st.write(f"- Time: {instance['stime']} - {instance['etime']}")
                                    st.write("---")
                        else:
                            st.success("No profanity detected from the agent.")

                        if borrower_profanity:
                            st.warning("Customer/Borrower Profanity Detected:")
                            for call_id, instances in borrower_profanity.items():
                                st.write(f"**Call ID: {call_id}**")
                                for i, instance in enumerate(instances):
                                    st.write(f"Instance {i + 1}:")
                                    st.write(f"- Text: *{instance['text']}*")
                                    st.write(f"- Profane words: {', '.join(instance['profane_words'])}")
                                    st.write(f"- Time: {instance['stime']} - {instance['etime']}")
                                    st.write("---")
                        else:
                            st.success("No profanity detected from the customer/borrower.")

                elif entity_type == "Privacy and Compliance Violation":
                    results = analyze_privacy_violation(conversations)
                    if results['privacy_violation']:
                        st.error("Privacy Violation Detected:")
                        for call_id, instances in results['privacy_violation'].items():
                            st.write(f"**Call ID: {call_id}**")
                            for i, instance in enumerate(instances):
                                st.write(f"Instance {i + 1}:")
                                st.write(f"- Text: *{instance['text']}*")
                                st.write(f"- Time: {instance['stime']} - {instance['etime']}")
                                st.write("---")
                    else:
                        st.success("No privacy violations detected.")

        else:
            st.error("Failed to load the conversation data. Please check the file format.")


if __name__ == "__main__":
    run_conversation_analysis_app()
