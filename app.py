
import re
import json
#import yaml
import os
import pandas as pd
import streamlit as st
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Download NLTK resources - fix for the punkt_tab error
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# =====================
# Profanity Detection - Pattern Matching Approach
# =====================

def load_profanity_list():
    """Load a list of profane words"""
    # This is a sample list for demonstration purposes
    # In a real implementation, you would use a more comprehensive list
    profane_words = [
        "fuck", "shit", "damn", "bitch", "asshole", "crap", "hell",
        "bastard", "dick", "piss", "cunt", "bullshit", "dumbass",
        "motherfucker", "goddamn", "ass", "whore", "slut"
    ]

    # Generate variations with common obfuscations
    variations = []
    for word in profane_words:
        variations.append(word)
        # Add some common obfuscations (e.g. sh*t, f*ck)
        for i in range(len(word)):
            if i > 0 and i < len(word) - 1:
                variations.append(word[:i] + '*' + word[i + 1:])
        # Add some common variations with numbers (e.g. a55, f0ck)
        variations.append(word.replace('a', '4').replace('e', '3').replace('i', '1').replace('o', '0'))

    return list(set(variations))


def detect_profanity_regex(text, profanity_list):
    """Detect profane language using regex pattern matching"""
    if not text or not isinstance(text, str):
        return False, []

    text = text.lower()
    found_words = []

    for word in profanity_list:
        # Using word boundaries to match whole words only
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
# Profanity Detection - Machine Learning Approach
# =====================

class MLProfanityDetector:
    """A simple ML-based profanity detector using word embeddings"""

    def __init__(self):
        """Initialize the ML profanity detector"""
        # For a real implementation, you would load a trained model here
        # This is a simplified version for demonstration
        self.profanity_list = load_profanity_list()

        # Additional contextual words that might indicate offensive content
        self.contextual_indicators = [
            "angry", "upset", "hate", "stupid", "idiot", "dumb",
            "annoying", "useless", "terrible", "horrible", "worst"
        ]

    def predict(self, text):
        """Predict if text contains profanity"""
        if not text or not isinstance(text, str):
            return False, []

        text = text.lower()

        # Simple string splitting instead of word_tokenize to avoid NLTK issues
        tokens = text.split()

        # First check for direct profanity
        found_words = []
        for word in self.profanity_list:
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text):
                found_words.append(word)

        # Then check for contextual indicators that might suggest offensive content
        contextual_score = sum(1 for word in tokens if word in self.contextual_indicators)

        # If we have direct profanity or strong contextual indicators
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
# Profanity Detection - LLM-based Approach
# =====================

def analyze_profanity_llm(conversations):
    """
    Simulate LLM-based profanity detection
    Note: In a real implementation, this would call an actual LLM API
    """
    results = {
        'agent_profanity': defaultdict(list),
        'borrower_profanity': defaultdict(list)
    }

    # For demonstration, we'll use the pattern matching approach as a substitute
    profanity_list = load_profanity_list()

    for call_id, conversation in conversations.items():
        for utterance in conversation:
            speaker = utterance.get('speaker', '').lower()
            text = utterance.get('text', '')

            # For demonstration, we'll just use our regex approach
            # In a real implementation, this would call the LLM API
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
# Compliance Detection - Pattern Matching Approach
# =====================

def load_sensitive_info_patterns():
    """Load patterns for sensitive information detection"""
    patterns = {
        'account_number': r'\b(?:account|acct\.?|a/?c)\s*(?:#|number|no\.?)?\s*[:#]?\s*(\d{4,})',
        'ssn': r'\b(?:SSN|social security|social security number)\s*(?:#|number|no\.?)?\s*[:#]?\s*(\d{3}-\d{2}-\d{4}|\d{9})',
        'balance': r'\b(?:balance|amount|sum|total)(?:\s+(?:due|owed|outstanding|unpaid))?\s*(?:is|of|:)?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        'date_of_birth': r'\b(?:DOB|date of birth|birth date|born on)\s*(?:is|:|-)?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        'address': r'\b(?:address|residence|live at|living at|located at)\s*(?:is|:|-)?\s*(\d+\s+[^\d]+(?:avenue|ave|street|st|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|circle|cir|trail|trl|way|place|pl)\b.{5,50})',
    }
    return patterns


def load_verification_patterns():
    """Load patterns for identity verification detection"""
    patterns = {
        'verification_question': r'\b(?:verify|confirm|validation|identify|authentication|prove)(?:\s+(?:your|identity|yourself|who you are))?\b',
        'dob_verification': r'\b(?:DOB|date of birth|birthday|born on|birth date)\b',
        'ssn_verification': r'\b(?:SSN|social security|last four|last 4)\b',
        'address_verification': r'\b(?:address|zip code|postal code|residence)\b',
        'verification_success': r'\b(?:thank you for verifying|verification successful|identity confirmed|that matches|that\'s correct|verification complete)\b'
    }
    return patterns


def detect_sensitive_information(text, patterns):
    """Detect sensitive information in text using regex patterns"""
    if not text or not isinstance(text, str):
        return {}, []

    text = text.lower()
    found_info = {}
    information_types = []

    for info_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            found_info[info_type] = matches
            information_types.append(info_type)

    return found_info, information_types


def detect_verification(text, patterns):
    """Detect identity verification in text using regex patterns"""
    if not text or not isinstance(text, str):
        return False, []

    text = text.lower()
    verification_types = []

    for verification_type, pattern in patterns.items():
        if re.search(pattern, text):
            verification_types.append(verification_type)

    # Consider verification present if we find any verification pattern
    return len(verification_types) > 0, verification_types


def analyze_compliance_pattern_matching(conversations):
    """
    Analyze conversations for compliance violations using pattern matching
    Returns calls where sensitive info was shared without proper verification
    """
    sensitive_patterns = load_sensitive_info_patterns()
    verification_patterns = load_verification_patterns()

    results = {
        'violations': defaultdict(list),
        'all_sensitive_info': defaultdict(list)
    }

    for call_id, conversation in conversations.items():
        # Track verification status throughout the call
        verification_done = False
        verification_types = []

        # Track sensitive information shared
        sensitive_info_shared = []

        # Process the conversation chronologically
        for utterance in conversation:
            speaker = utterance.get('speaker', '').lower()
            text = utterance.get('text', '')

            # Check for verification (can be done by agent or confirmed by borrower)
            is_verification, found_verification = detect_verification(text, verification_patterns)
            if is_verification:
                verification_done = True
                verification_types.extend(found_verification)

            # Check for sensitive information in agent utterances
            if 'agent' in speaker:
                sensitive_info, info_types = detect_sensitive_information(text, sensitive_patterns)

                if sensitive_info:
                    # Record all instances of sensitive information
                    results['all_sensitive_info'][call_id].append({
                        'text': text,
                        'sensitive_info': sensitive_info,
                        'stime': utterance.get('stime'),
                        'etime': utterance.get('etime')
                    })

                    # If sensitive info shared before verification, record violation
                    if not verification_done and sensitive_info:
                        results['violations'][call_id].append({
                            'text': text,
                            'sensitive_info': sensitive_info,
                            'verification_status': 'Not verified',
                            'stime': utterance.get('stime'),
                            'etime': utterance.get('etime')
                        })

    return results


# =====================
# Compliance Detection - ML-based Approach
# =====================

class ComplianceMLDetector:
    """ML-based detector for compliance violations"""

    def __init__(self):
        """Initialize the ML-based compliance detector"""
        # In a real implementation, you would load a trained model here
        # This is a simplified version for demonstration
        self.sensitive_patterns = load_sensitive_info_patterns()
        self.verification_patterns = load_verification_patterns()

        # Additional contextual phrases for better detection
        self.sensitive_info_indicators = [
            "your account", "your balance", "you owe", "outstanding amount",
            "personal information", "payment details", "account status"
        ]

        self.verification_indicators = [
            "can you confirm", "need to verify", "for security purposes",
            "before we proceed", "to protect your privacy", "authenticate"
        ]

    def detect_sensitive_information(self, text):
        """ML-enhanced detection of sensitive information"""
        # First use regex patterns
        found_info, info_types = detect_sensitive_information(text, self.sensitive_patterns)

        # Then enhance with contextual understanding
        text_lower = text.lower()
        for indicator in self.sensitive_info_indicators:
            if indicator in text_lower and not info_types:
                # Mark as potentially containing implicit sensitive information
                info_types.append("implicit_sensitive_info")
                found_info["implicit_sensitive_info"] = ["contextual reference"]
                break

        return found_info, info_types

    def detect_verification(self, text):
        """ML-enhanced detection of verification"""
        # First use regex patterns
        is_verification, verification_types = detect_verification(text, self.verification_patterns)

        # Then enhance with contextual understanding
        if not is_verification:
            text_lower = text.lower()
            for indicator in self.verification_indicators:
                if indicator in text_lower:
                    verification_types.append("implicit_verification")
                    is_verification = True
                    break

        return is_verification, verification_types


def analyze_compliance_ml(conversations):
    """
    Analyze conversations for compliance violations using ML approach
    Returns calls where sensitive info was shared without proper verification
    """
    detector = ComplianceMLDetector()

    results = {
        'violations': defaultdict(list),
        'all_sensitive_info': defaultdict(list)
    }

    for call_id, conversation in conversations.items():
        # Track verification status throughout the call
        verification_done = False
        verification_types = []

        # Track sensitive information shared
        sensitive_info_shared = []

        # Process the conversation chronologically
        for utterance in conversation:
            speaker = utterance.get('speaker', '').lower()
            text = utterance.get('text', '')

            # Check for verification (can be done by agent or confirmed by borrower)
            is_verification, found_verification = detector.detect_verification(text)
            if is_verification:
                verification_done = True
                verification_types.extend(found_verification)

            # Check for sensitive information in agent utterances
            if 'agent' in speaker:
                sensitive_info, info_types = detector.detect_sensitive_information(text)

                if sensitive_info:
                    # Record all instances of sensitive information
                    results['all_sensitive_info'][call_id].append({
                        'text': text,
                        'sensitive_info': sensitive_info,
                        'stime': utterance.get('stime'),
                        'etime': utterance.get('etime')
                    })

                    # If sensitive info shared before verification, record violation
                    if not verification_done and sensitive_info:
                        results['violations'][call_id].append({
                            'text': text,
                            'sensitive_info': sensitive_info,
                            'verification_status': 'Not verified',
                            'stime': utterance.get('stime'),
                            'etime': utterance.get('etime')
                        })

    return results


# =====================
# Compliance Detection - LLM-based Approach
# =====================

def analyze_compliance_llm(conversations):
    """
    Simulate LLM-based compliance violation detection
    Note: In a real implementation, this would call an actual LLM API
    """
    # For demonstration purposes, we'll use our pattern matching implementation
    # In a real-world scenario, this would call an LLM API with appropriate prompting
    return analyze_compliance_pattern_matching(conversations)


# =====================
# Data Loading Functions
# =====================

def load_data(file_content):
    """Load conversation data from a JSON string or file"""
    try:
        if isinstance(file_content, str):
            # Try to parse as JSON
            return json.loads(file_content)
        else:
            # Read the file and parse as JSON
            data = json.load(file_content)
            return data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# =====================
# Visualization Functions
# =====================

def create_profanity_chart(results):
    """Create a chart showing profanity statistics"""
    # Prepare data for visualization
    agent_counts = {}
    borrower_counts = {}

    for call_id, instances in results['agent_profanity'].items():
        agent_counts[call_id] = len(instances)

    for call_id, instances in results['borrower_profanity'].items():
        borrower_counts[call_id] = len(instances)

    # Convert to DataFrame
    all_call_ids = set(list(agent_counts.keys()) + list(borrower_counts.keys()))
    data = []

    for call_id in all_call_ids:
        data.append({
            'Call ID': call_id,
            'Agent Profanity Instances': agent_counts.get(call_id, 0),
            'Borrower Profanity Instances': borrower_counts.get(call_id, 0)
        })

    df = pd.DataFrame(data)

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(x='Call ID', kind='bar', stacked=False, ax=ax)
    plt.title('Profanity Instances by Call')
    plt.ylabel('Number of Instances')
    plt.tight_layout()

    return fig


def create_compliance_chart(results):
    """Create a chart showing compliance violation statistics"""
    # Prepare data for visualization
    violation_counts = {}
    sensitive_info_counts = {}

    for call_id, instances in results['violations'].items():
        violation_counts[call_id] = len(instances)

    for call_id, instances in results['all_sensitive_info'].items():
        sensitive_info_counts[call_id] = len(instances)

    # Convert to DataFrame
    all_call_ids = set(list(violation_counts.keys()) + list(sensitive_info_counts.keys()))
    data = []

    for call_id in all_call_ids:
        data.append({
            'Call ID': call_id,
            'Compliance Violations': violation_counts.get(call_id, 0),
            'Sensitive Info Instances': sensitive_info_counts.get(call_id, 0)
        })

    df = pd.DataFrame(data)

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(x='Call ID', kind='bar', stacked=False, ax=ax)
    plt.title('Compliance Statistics by Call')
    plt.ylabel('Number of Instances')
    plt.tight_layout()

    return fig


# =====================
# Streamlit App
# =====================

def run_call_analysis_app():
    st.title("Debt Collection Call Analysis")

    # File upload
    uploaded_file = st.file_uploader("Upload a conversation file (JSON)", type=["json"])

    # Two columns for configuration
    col1, col2 = st.columns(2)

    with col1:
        # Select entity type
        entity_type = st.selectbox(
            "Select Analysis Type",
            ["Profanity Detection", "Privacy and Compliance Violation"]
        )

    with col2:
        # Select approach
        approach = st.selectbox(
            "Select Analysis Approach",
            ["Pattern Matching", "Machine Learning", "LLM-based"]
        )

    if uploaded_file is not None:
        # Load the data
        conversation_data = load_data(uploaded_file)

        if conversation_data:
            # For the sample format, wrap in a dictionary with a generated call ID
            call_id = uploaded_file.name.split('.')[0]
            conversations = {call_id: conversation_data}

            if st.button("Analyze"):
                st.subheader("Analysis Results")

                if entity_type == "Profanity Detection":
                    with st.spinner("Analyzing for profanity..."):
                        if approach == "Pattern Matching":
                            results = analyze_profanity_pattern_matching(conversations)
                        elif approach == "Machine Learning":
                            results = analyze_profanity_ml(conversations)
                        else:  # LLM-based
                            results = analyze_profanity_llm(conversations)

                        agent_profanity = results['agent_profanity']
                        borrower_profanity = results['borrower_profanity']

                        # Display visualization
                        if agent_profanity or borrower_profanity:
                            st.subheader("Profanity Statistics")
                            fig = create_profanity_chart(results)
                            st.pyplot(fig)

                        # Display results
                        if not agent_profanity and not borrower_profanity:
                            st.success("No profanity detected in the conversation.")
                        else:
                            # Display agent profanity
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

                            # Display borrower profanity
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
                else:  # Privacy and Compliance Violation
                    with st.spinner("Analyzing for compliance violations..."):
                        if approach == "Pattern Matching":
                            results = analyze_compliance_pattern_matching(conversations)
                        elif approach == "Machine Learning":
                            results = analyze_compliance_ml(conversations)
                        else:  # LLM-based
                            results = analyze_compliance_llm(conversations)

                        violations = results['violations']
                        all_sensitive_info = results['all_sensitive_info']

                        # Display visualization
                        if violations or all_sensitive_info:
                            st.subheader("Compliance Violation Statistics")
                            fig = create_compliance_chart(results)
                            st.pyplot(fig)

                        # Display results
                        if not violations:
                            st.success("No compliance violations detected.")
                        else:
                            st.error("Compliance Violations Detected:")
                            for call_id, instances in violations.items():
                                st.write(f"**Call ID: {call_id}**")
                                for i, instance in enumerate(instances):
                                    st.write(f"Violation {i + 1}:")
                                    st.write(f"- Text: *{instance['text']}*")
                                    st.write(f"- Sensitive info shared: {', '.join(instance['sensitive_info'].keys())}")
                                    st.write(f"- Verification status: {instance['verification_status']}")
                                    st.write(f"- Time: {instance['stime']} - {instance['etime']}")
                                    st.write("---")

                        # Display all sensitive information for reference
                        st.subheader("All Sensitive Information")
                        if not all_sensitive_info:
                            st.info("No sensitive information detected.")
                        else:
                            for call_id, instances in all_sensitive_info.items():
                                st.write(f"**Call ID: {call_id}**")
                                for i, instance in enumerate(instances):
                                    st.write(f"Instance {i + 1}:")
                                    st.write(f"- Text: *{instance['text']}*")
                                    st.write(f"- Info types: {', '.join(instance['sensitive_info'].keys())}")
                                    st.write(f"- Time: {instance['stime']} - {instance['etime']}")
                                    st.write("---")
        else:
            st.error("Failed to load the conversation data. Please check the file format.")


if __name__ == "__main__":
    run_call_analysis_app()



# import re
# import json
# #import yaml
# import os
# import pandas as pd
# import streamlit as st
# from collections import defaultdict
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # Download NLTK resources
# nltk.download('punkt', quiet=True)
#
#
# # =====================
# # Profanity Detection - Pattern Matching Approach
# # =====================
#
# def load_profanity_list():
#     """Load a list of profane words"""
#     # This is a sample list for demonstration purposes
#     # In a real implementation, you would use a more comprehensive list
#     profane_words = [
#         "fuck", "shit", "damn", "bitch", "asshole", "crap", "hell",
#         "bastard", "dick", "piss", "cunt", "bullshit", "dumbass",
#         "motherfucker", "goddamn", "ass", "whore", "slut"
#     ]
#
#     # Generate variations with common obfuscations
#     variations = []
#     for word in profane_words:
#         variations.append(word)
#         # Add some common obfuscations (e.g. sh*t, f*ck)
#         for i in range(len(word)):
#             if i > 0 and i < len(word) - 1:
#                 variations.append(word[:i] + '*' + word[i + 1:])
#         # Add some common variations with numbers (e.g. a55, f0ck)
#         variations.append(word.replace('a', '4').replace('e', '3').replace('i', '1').replace('o', '0'))
#
#     return list(set(variations))
#
#
# def detect_profanity_regex(text, profanity_list):
#     """Detect profane language using regex pattern matching"""
#     if not text or not isinstance(text, str):
#         return False, []
#
#     text = text.lower()
#     found_words = []
#
#     for word in profanity_list:
#         # Using word boundaries to match whole words only
#         pattern = r'\b' + re.escape(word) + r'\b'
#         if re.search(pattern, text):
#             found_words.append(word)
#
#     return len(found_words) > 0, found_words
#
#
# def analyze_profanity_pattern_matching(conversations):
#     """Analyze all conversations for profanity using pattern matching"""
#     profanity_list = load_profanity_list()
#     results = {
#         'agent_profanity': defaultdict(list),
#         'borrower_profanity': defaultdict(list)
#     }
#
#     for call_id, conversation in conversations.items():
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             has_profanity, found_words = detect_profanity_regex(text, profanity_list)
#
#             if has_profanity:
#                 if 'agent' in speaker:
#                     results['agent_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#                 elif any(term in speaker.lower() for term in ['customer', 'borrower']):
#                     results['borrower_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#     return results
#
#
# # =====================
# # Profanity Detection - Machine Learning Approach
# # =====================
#
# class MLProfanityDetector:
#     """A simple ML-based profanity detector using word embeddings"""
#
#     def __init__(self):
#         """Initialize the ML profanity detector"""
#         # For a real implementation, you would load a trained model here
#         # This is a simplified version for demonstration
#         self.profanity_list = load_profanity_list()
#
#         # Additional contextual words that might indicate offensive content
#         self.contextual_indicators = [
#             "angry", "upset", "hate", "stupid", "idiot", "dumb",
#             "annoying", "useless", "terrible", "horrible", "worst"
#         ]
#
#     def predict(self, text):
#         """Predict if text contains profanity"""
#         if not text or not isinstance(text, str):
#             return False, []
#
#         text = text.lower()
#         tokens = word_tokenize(text)
#
#         # First check for direct profanity
#         found_words = []
#         for word in self.profanity_list:
#             pattern = r'\b' + re.escape(word) + r'\b'
#             if re.search(pattern, text):
#                 found_words.append(word)
#
#         # Then check for contextual indicators that might suggest offensive content
#         contextual_score = sum(1 for word in tokens if word in self.contextual_indicators)
#
#         # If we have direct profanity or strong contextual indicators
#         if found_words or contextual_score >= 2:
#             return True, found_words if found_words else ["contextual offensive language"]
#         else:
#             return False, []
#
#
# def analyze_profanity_ml(conversations):
#     """Analyze all conversations for profanity using the ML approach"""
#     detector = MLProfanityDetector()
#     results = {
#         'agent_profanity': defaultdict(list),
#         'borrower_profanity': defaultdict(list)
#     }
#
#     for call_id, conversation in conversations.items():
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             has_profanity, found_words = detector.predict(text)
#
#             if has_profanity:
#                 if 'agent' in speaker:
#                     results['agent_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#                 elif any(term in speaker.lower() for term in ['customer', 'borrower']):
#                     results['borrower_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#     return results
#
#
# # =====================
# # Profanity Detection - LLM-based Approach
# # =====================
#
# def analyze_profanity_llm(conversations):
#     """
#     Simulate LLM-based profanity detection
#     Note: In a real implementation, this would call an actual LLM API
#     """
#     results = {
#         'agent_profanity': defaultdict(list),
#         'borrower_profanity': defaultdict(list)
#     }
#
#     # For demonstration, we'll use the pattern matching approach as a substitute
#     profanity_list = load_profanity_list()
#
#     for call_id, conversation in conversations.items():
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             # For demonstration, we'll just use our regex approach
#             # In a real implementation, this would call the LLM API
#             has_profanity, found_words = detect_profanity_regex(text, profanity_list)
#
#             if has_profanity:
#                 if 'agent' in speaker:
#                     results['agent_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#                 elif any(term in speaker.lower() for term in ['customer', 'borrower']):
#                     results['borrower_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#     return results
#
#
# # =====================
# # Compliance Detection - Pattern Matching Approach
# # =====================
#
# def load_sensitive_info_patterns():
#     """Load patterns for sensitive information detection"""
#     patterns = {
#         'account_number': r'\b(?:account|acct\.?|a/?c)\s*(?:#|number|no\.?)?\s*[:#]?\s*(\d{4,})',
#         'ssn': r'\b(?:SSN|social security|social security number)\s*(?:#|number|no\.?)?\s*[:#]?\s*(\d{3}-\d{2}-\d{4}|\d{9})',
#         'balance': r'\b(?:balance|amount|sum|total)(?:\s+(?:due|owed|outstanding|unpaid))?\s*(?:is|of|:)?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
#         'date_of_birth': r'\b(?:DOB|date of birth|birth date|born on)\s*(?:is|:|-)?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
#         'address': r'\b(?:address|residence|live at|living at|located at)\s*(?:is|:|-)?\s*(\d+\s+[^\d]+(?:avenue|ave|street|st|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|circle|cir|trail|trl|way|place|pl)\b.{5,50})',
#     }
#     return patterns
#
#
# def load_verification_patterns():
#     """Load patterns for identity verification detection"""
#     patterns = {
#         'verification_question': r'\b(?:verify|confirm|validation|identify|authentication|prove)(?:\s+(?:your|identity|yourself|who you are))?\b',
#         'dob_verification': r'\b(?:DOB|date of birth|birthday|born on|birth date)\b',
#         'ssn_verification': r'\b(?:SSN|social security|last four|last 4)\b',
#         'address_verification': r'\b(?:address|zip code|postal code|residence)\b',
#         'verification_success': r'\b(?:thank you for verifying|verification successful|identity confirmed|that matches|thats correct | verification complete)\b'
#     }
#     return patterns
#
#
# def detect_sensitive_information(text, patterns):
#     """Detect sensitive information in text using regex patterns"""
#     if not text or not isinstance(text, str):
#         return {}, []
#
#     text = text.lower()
#     found_info = {}
#     information_types = []
#
#     for info_type, pattern in patterns.items():
#         matches = re.findall(pattern, text)
#         if matches:
#             found_info[info_type] = matches
#             information_types.append(info_type)
#
#     return found_info, information_types
#
#
# def detect_verification(text, patterns):
#     """Detect identity verification in text using regex patterns"""
#     if not text or not isinstance(text, str):
#         return False, []
#
#     text = text.lower()
#     verification_types = []
#
#     for verification_type, pattern in patterns.items():
#         if re.search(pattern, text):
#             verification_types.append(verification_type)
#
#     # Consider verification present if we find any verification pattern
#     return len(verification_types) > 0, verification_types
#
#
# def analyze_compliance_pattern_matching(conversations):
#     """
#     Analyze conversations for compliance violations using pattern matching
#     Returns calls where sensitive info was shared without proper verification
#     """
#     sensitive_patterns = load_sensitive_info_patterns()
#     verification_patterns = load_verification_patterns()
#
#     results = {
#         'violations': defaultdict(list),
#         'all_sensitive_info': defaultdict(list)
#     }
#
#     for call_id, conversation in conversations.items():
#         # Track verification status throughout the call
#         verification_done = False
#         verification_types = []
#
#         # Track sensitive information shared
#         sensitive_info_shared = []
#
#         # Process the conversation chronologically
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             # Check for verification (can be done by agent or confirmed by borrower)
#             is_verification, found_verification = detect_verification(text, verification_patterns)
#             if is_verification:
#                 verification_done = True
#                 verification_types.extend(found_verification)
#
#             # Check for sensitive information in agent utterances
#             if 'agent' in speaker:
#                 sensitive_info, info_types = detect_sensitive_information(text, sensitive_patterns)
#
#                 if sensitive_info:
#                     # Record all instances of sensitive information
#                     results['all_sensitive_info'][call_id].append({
#                         'text': text,
#                         'sensitive_info': sensitive_info,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#                     # If sensitive info shared before verification, record violation
#                     if not verification_done and sensitive_info:
#                         results['violations'][call_id].append({
#                             'text': text,
#                             'sensitive_info': sensitive_info,
#                             'verification_status': 'Not verified',
#                             'stime': utterance.get('stime'),
#                             'etime': utterance.get('etime')
#                         })
#
#     return results
#
#
# # =====================
# # Compliance Detection - ML-based Approach
# # =====================
#
# class ComplianceMLDetector:
#     """ML-based detector for compliance violations"""
#
#     def __init__(self):
#         """Initialize the ML-based compliance detector"""
#         # In a real implementation, you would load a trained model here
#         # This is a simplified version for demonstration
#         self.sensitive_patterns = load_sensitive_info_patterns()
#         self.verification_patterns = load_verification_patterns()
#
#         # Additional contextual phrases for better detection
#         self.sensitive_info_indicators = [
#             "your account", "your balance", "you owe", "outstanding amount",
#             "personal information", "payment details", "account status"
#         ]
#
#         self.verification_indicators = [
#             "can you confirm", "need to verify", "for security purposes",
#             "before we proceed", "to protect your privacy", "authenticate"
#         ]
#
#     def detect_sensitive_information(self, text):
#         """ML-enhanced detection of sensitive information"""
#         # First use regex patterns
#         found_info, info_types = detect_sensitive_information(text, self.sensitive_patterns)
#
#         # Then enhance with contextual understanding
#         text_lower = text.lower()
#         for indicator in self.sensitive_info_indicators:
#             if indicator in text_lower and not info_types:
#                 # Mark as potentially containing implicit sensitive information
#                 info_types.append("implicit_sensitive_info")
#                 found_info["implicit_sensitive_info"] = ["contextual reference"]
#                 break
#
#         return found_info, info_types
#
#     def detect_verification(self, text):
#         """ML-enhanced detection of verification"""
#         # First use regex patterns
#         is_verification, verification_types = detect_verification(text, self.verification_patterns)
#
#         # Then enhance with contextual understanding
#         if not is_verification:
#             text_lower = text.lower()
#             for indicator in self.verification_indicators:
#                 if indicator in text_lower:
#                     verification_types.append("implicit_verification")
#                     is_verification = True
#                     break
#
#         return is_verification, verification_types
#
#
# def analyze_compliance_ml(conversations):
#     """
#     Analyze conversations for compliance violations using ML approach
#     Returns calls where sensitive info was shared without proper verification
#     """
#     detector = ComplianceMLDetector()
#
#     results = {
#         'violations': defaultdict(list),
#         'all_sensitive_info': defaultdict(list)
#     }
#
#     for call_id, conversation in conversations.items():
#         # Track verification status throughout the call
#         verification_done = False
#         verification_types = []
#
#         # Track sensitive information shared
#         sensitive_info_shared = []
#
#         # Process the conversation chronologically
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             # Check for verification (can be done by agent or confirmed by borrower)
#             is_verification, found_verification = detector.detect_verification(text)
#             if is_verification:
#                 verification_done = True
#                 verification_types.extend(found_verification)
#
#             # Check for sensitive information in agent utterances
#             if 'agent' in speaker:
#                 sensitive_info, info_types = detector.detect_sensitive_information(text)
#
#                 if sensitive_info:
#                     # Record all instances of sensitive information
#                     results['all_sensitive_info'][call_id].append({
#                         'text': text,
#                         'sensitive_info': sensitive_info,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#                     # If sensitive info shared before verification, record violation
#                     if not verification_done and sensitive_info:
#                         results['violations'][call_id].append({
#                             'text': text,
#                             'sensitive_info': sensitive_info,
#                             'verification_status': 'Not verified',
#                             'stime': utterance.get('stime'),
#                             'etime': utterance.get('etime')
#                         })
#
#     return results
#
#
# # =====================
# # Compliance Detection - LLM-based Approach
# # =====================
#
# def analyze_compliance_llm(conversations):
#     """
#     Simulate LLM-based compliance violation detection
#     Note: In a real implementation, this would call an actual LLM API
#     """
#     # For demonstration purposes, we'll use our pattern matching implementation
#     # In a real-world scenario, this would call an LLM API with appropriate prompting
#     return analyze_compliance_pattern_matching(conversations)
#
#
# # =====================
# # Data Loading Functions
# # =====================
#
# def load_data(file_content):
#     """Load conversation data from a JSON string or file"""
#     try:
#         if isinstance(file_content, str):
#             # Try to parse as JSON
#             return json.loads(file_content)
#         else:
#             # Read the file and parse as JSON
#             data = json.load(file_content)
#             return data
#     except json.JSONDecodeError as e:
#         st.error(f"Error parsing JSON: {e}")
#         return None
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return None
#
#
# # =====================
# # Visualization Functions
# # =====================
#
# def create_profanity_chart(results):
#     """Create a chart showing profanity statistics"""
#     # Prepare data for visualization
#     agent_counts = {}
#     borrower_counts = {}
#
#     for call_id, instances in results['agent_profanity'].items():
#         agent_counts[call_id] = len(instances)
#
#     for call_id, instances in results['borrower_profanity'].items():
#         borrower_counts[call_id] = len(instances)
#
#     # Convert to DataFrame
#     all_call_ids = set(list(agent_counts.keys()) + list(borrower_counts.keys()))
#     data = []
#
#     for call_id in all_call_ids:
#         data.append({
#             'Call ID': call_id,
#             'Agent Profanity Instances': agent_counts.get(call_id, 0),
#             'Borrower Profanity Instances': borrower_counts.get(call_id, 0)
#         })
#
#     df = pd.DataFrame(data)
#
#     # Create chart
#     fig, ax = plt.subplots(figsize=(10, 6))
#     df.plot(x='Call ID', kind='bar', stacked=False, ax=ax)
#     plt.title('Profanity Instances by Call')
#     plt.ylabel('Number of Instances')
#     plt.tight_layout()
#
#     return fig
#
#
# def create_compliance_chart(results):
#     """Create a chart showing compliance violation statistics"""
#     # Prepare data for visualization
#     violation_counts = {}
#     sensitive_info_counts = {}
#
#     for call_id, instances in results['violations'].items():
#         violation_counts[call_id] = len(instances)
#
#     for call_id, instances in results['all_sensitive_info'].items():
#         sensitive_info_counts[call_id] = len(instances)
#
#     # Convert to DataFrame
#     all_call_ids = set(list(violation_counts.keys()) + list(sensitive_info_counts.keys()))
#     data = []
#
#     for call_id in all_call_ids:
#         data.append({
#             'Call ID': call_id,
#             'Compliance Violations': violation_counts.get(call_id, 0),
#             'Sensitive Info Instances': sensitive_info_counts.get(call_id, 0)
#         })
#
#     df = pd.DataFrame(data)
#
#     # Create chart
#     fig, ax = plt.subplots(figsize=(10, 6))
#     df.plot(x='Call ID', kind='bar', stacked=False, ax=ax)
#     plt.title('Compliance Statistics by Call')
#     plt.ylabel('Number of Instances')
#     plt.tight_layout()
#
#     return fig
#
#
# # =====================
# # Streamlit App
# # =====================
#
# def run_call_analysis_app():
#     st.title("Debt Collection Call Analysis")
#
#     # File upload
#     uploaded_file = st.file_uploader("Upload a conversation file (JSON)", type=["json"])
#
#     # Two columns for configuration
#     col1, col2 = st.columns(2)
#
#     with col1:
#         # Select entity type
#         entity_type = st.selectbox(
#             "Select Analysis Type",
#             ["Profanity Detection", "Privacy and Compliance Violation"]
#         )
#
#     with col2:
#         # Select approach
#         approach = st.selectbox(
#             "Select Analysis Approach",
#             ["Pattern Matching", "Machine Learning", "LLM-based"]
#         )
#
#     if uploaded_file is not None:
#         # Load the data
#         conversation_data = load_data(uploaded_file)
#
#         if conversation_data:
#             # For the sample format, wrap in a dictionary with a generated call ID
#             call_id = uploaded_file.name.split('.')[0]
#             conversations = {call_id: conversation_data}
#
#             if st.button("Analyze"):
#                 st.subheader("Analysis Results")
#
#                 if entity_type == "Profanity Detection":
#                     with st.spinner("Analyzing for profanity..."):
#                         if approach == "Pattern Matching":
#                             results = analyze_profanity_pattern_matching(conversations)
#                         elif approach == "Machine Learning":
#                             results = analyze_profanity_ml(conversations)
#                         else:  # LLM-based
#                             results = analyze_profanity_llm(conversations)
#
#                         agent_profanity = results['agent_profanity']
#                         borrower_profanity = results['borrower_profanity']
#
#                         # Display visualization
#                         if agent_profanity or borrower_profanity:
#                             st.subheader("Profanity Statistics")
#                             fig = create_profanity_chart(results)
#                             st.pyplot(fig)
#
#                         # Display results
#                         if not agent_profanity and not borrower_profanity:
#                             st.success("No profanity detected in the conversation.")
#                         else:
#                             # Display agent profanity
#                             if agent_profanity:
#                                 st.error("Agent Profanity Detected:")
#                                 for call_id, instances in agent_profanity.items():
#                                     st.write(f"**Call ID: {call_id}**")
#                                     for i, instance in enumerate(instances):
#                                         st.write(f"Instance {i + 1}:")
#                                         st.write(f"- Text: *{instance['text']}*")
#                                         st.write(f"- Profane words: {', '.join(instance['profane_words'])}")
#                                         st.write(f"- Time: {instance['stime']} - {instance['etime']}")
#                                         st.write("---")
#                             else:
#                                 st.success("No profanity detected from the agent.")
#
#                             # Display borrower profanity
#                             if borrower_profanity:
#                                 st.warning("Customer/Borrower Profanity Detected:")
#                                 for call_id, instances in borrower_profanity.items():
#                                     st.write(f"**Call ID: {call_id}**")
#                                     for i, instance in enumerate(instances):
#                                         st.write(f"Instance {i + 1}:")
#                                         st.write(f"- Text: *{instance['text']}*")
#                                         st.write(f"- Profane words: {', '.join(instance['profane_words'])}")
#                                         st.write(f"- Time: {instance['stime']} - {instance['etime']}")
#                                         st.write("---")
#                             else:
#                                 st.success("No profanity detected from the customer/borrower.")
#                 else:  # Privacy and Compliance Violation
#                     with st.spinner("Analyzing for compliance violations..."):
#                         if approach == "Pattern Matching":
#                             results = analyze_compliance_pattern_matching(conversations)
#                         elif approach == "Machine Learning":
#                             results = analyze_compliance_ml(conversations)
#                         else:  # LLM-based
#                             results = analyze_compliance_llm(conversations)
#
#                         violations = results['violations']
#                         all_sensitive_info = results['all_sensitive_info']
#
#                         # Display visualization
#                         if violations or all_sensitive_info:
#                             st.subheader("Compliance Violation Statistics")
#                             fig = create_compliance_chart(results)
#                             st.pyplot(fig)
#
#                         # Display results
#                         if not violations:
#                             st.success("No compliance violations detected.")
#                         else:
#                             st.error("Compliance Violations Detected:")
#                             for call_id, instances in violations.items():
#                                 st.write(f"**Call ID: {call_id}**")
#                                 for i, instance in enumerate(instances):
#                                     st.write(f"Violation {i + 1}:")
#                                     st.write(f"- Text: *{instance['text']}*")
#                                     st.write(f"- Sensitive info shared: {', '.join(instance['sensitive_info'].keys())}")
#                                     st.write(f"- Verification status: {instance['verification_status']}")
#                                     st.write(f"- Time: {instance['stime']} - {instance['etime']}")
#                                     st.write("---")
#
#                         # Display all sensitive information for reference
#                         st.subheader("All Sensitive Information")
#                         if not all_sensitive_info:
#                             st.info("No sensitive information detected.")
#                         else:
#                             for call_id, instances in all_sensitive_info.items():
#                                 st.write(f"**Call ID: {call_id}**")
#                                 for i, instance in enumerate(instances):
#                                     st.write(f"Instance {i + 1}:")
#                                     st.write(f"- Text: *{instance['text']}*")
#                                     st.write(f"- Info types: {', '.join(instance['sensitive_info'].keys())}")
#                                     st.write(f"- Time: {instance['stime']} - {instance['etime']}")
#                                     st.write("---")
#         else:
#             st.error("Failed to load the conversation data. Please check the file format.")
#
#
# if __name__ == "__main__":
#     run_call_analysis_app()


# import re
# import json
# #import yaml
# import os
# import pandas as pd
# import streamlit as st
# from collections import defaultdict
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # Download NLTK resources - fix for the punkt_tab error
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
#
#
# # =====================
# # Profanity Detection - Pattern Matching Approach
# # =====================
#
# def load_profanity_list():
#     """Load a list of profane words"""
#     # This is a sample list for demonstration purposes
#     # In a real implementation, you would use a more comprehensive list
#     profane_words = [
#         "fuck", "shit", "damn", "bitch", "asshole", "crap", "hell",
#         "bastard", "dick", "piss", "cunt", "bullshit", "dumbass",
#         "motherfucker", "goddamn", "ass", "whore", "slut"
#     ]
#
#     # Generate variations with common obfuscations
#     variations = []
#     for word in profane_words:
#         variations.append(word)
#         # Add some common obfuscations (e.g. sh*t, f*ck)
#         for i in range(len(word)):
#             if i > 0 and i < len(word) - 1:
#                 variations.append(word[:i] + '*' + word[i + 1:])
#         # Add some common variations with numbers (e.g. a55, f0ck)
#         variations.append(word.replace('a', '4').replace('e', '3').replace('i', '1').replace('o', '0'))
#
#     return list(set(variations))
#
#
# def detect_profanity_regex(text, profanity_list):
#     """Detect profane language using regex pattern matching"""
#     if not text or not isinstance(text, str):
#         return False, []
#
#     text = text.lower()
#     found_words = []
#
#     for word in profanity_list:
#         # Using word boundaries to match whole words only
#         pattern = r'\b' + re.escape(word) + r'\b'
#         if re.search(pattern, text):
#             found_words.append(word)
#
#     return len(found_words) > 0, found_words
#
#
# def analyze_profanity_pattern_matching(conversations):
#     """Analyze all conversations for profanity using pattern matching"""
#     profanity_list = load_profanity_list()
#     results = {
#         'agent_profanity': defaultdict(list),
#         'borrower_profanity': defaultdict(list)
#     }
#
#     for call_id, conversation in conversations.items():
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             has_profanity, found_words = detect_profanity_regex(text, profanity_list)
#
#             if has_profanity:
#                 if 'agent' in speaker:
#                     results['agent_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#                 elif any(term in speaker.lower() for term in ['customer', 'borrower']):
#                     results['borrower_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#     return results
#
#
# # =====================
# # Profanity Detection - Machine Learning Approach
# # =====================
#
# class MLProfanityDetector:
#     """A simple ML-based profanity detector using word embeddings"""
#
#     def __init__(self):
#         """Initialize the ML profanity detector"""
#         # For a real implementation, you would load a trained model here
#         # This is a simplified version for demonstration
#         self.profanity_list = load_profanity_list()
#
#         # Additional contextual words that might indicate offensive content
#         self.contextual_indicators = [
#             "angry", "upset", "hate", "stupid", "idiot", "dumb",
#             "annoying", "useless", "terrible", "horrible", "worst"
#         ]
#
#     def predict(self, text):
#         """Predict if text contains profanity"""
#         if not text or not isinstance(text, str):
#             return False, []
#
#         text = text.lower()
#
#         # Simple string splitting instead of word_tokenize to avoid NLTK issues
#         tokens = text.split()
#
#         # First check for direct profanity
#         found_words = []
#         for word in self.profanity_list:
#             pattern = r'\b' + re.escape(word) + r'\b'
#             if re.search(pattern, text):
#                 found_words.append(word)
#
#         # Then check for contextual indicators that might suggest offensive content
#         contextual_score = sum(1 for word in tokens if word in self.contextual_indicators)
#
#         # If we have direct profanity or strong contextual indicators
#         if found_words or contextual_score >= 2:
#             return True, found_words if found_words else ["contextual offensive language"]
#         else:
#             return False, []
#
#
# def analyze_profanity_ml(conversations):
#     """Analyze all conversations for profanity using the ML approach"""
#     detector = MLProfanityDetector()
#     results = {
#         'agent_profanity': defaultdict(list),
#         'borrower_profanity': defaultdict(list)
#     }
#
#     for call_id, conversation in conversations.items():
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             has_profanity, found_words = detector.predict(text)
#
#             if has_profanity:
#                 if 'agent' in speaker:
#                     results['agent_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#                 elif any(term in speaker.lower() for term in ['customer', 'borrower']):
#                     results['borrower_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#     return results
#
#
# # =====================
# # Profanity Detection - LLM-based Approach
# # =====================
#
# def analyze_profanity_llm(conversations):
#     """
#     Simulate LLM-based profanity detection
#     Note: In a real implementation, this would call an actual LLM API
#     """
#     results = {
#         'agent_profanity': defaultdict(list),
#         'borrower_profanity': defaultdict(list)
#     }
#
#     # For demonstration, we'll use the pattern matching approach as a substitute
#     profanity_list = load_profanity_list()
#
#     for call_id, conversation in conversations.items():
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             # For demonstration, we'll just use our regex approach
#             # In a real implementation, this would call the LLM API
#             has_profanity, found_words = detect_profanity_regex(text, profanity_list)
#
#             if has_profanity:
#                 if 'agent' in speaker:
#                     results['agent_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#                 elif any(term in speaker.lower() for term in ['customer', 'borrower']):
#                     results['borrower_profanity'][call_id].append({
#                         'text': text,
#                         'profane_words': found_words,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#     return results
#
#
# # =====================
# # Compliance Detection - Pattern Matching Approach
# # =====================
#
# def load_sensitive_info_patterns():
#     """Load patterns for sensitive information detection"""
#     patterns = {
#         'account_number': r'\b(?:account|acct\.?|a/?c)\s*(?:#|number|no\.?)?\s*[:#]?\s*(\d{4,})',
#         'ssn': r'\b(?:SSN|social security|social security number)\s*(?:#|number|no\.?)?\s*[:#]?\s*(\d{3}-\d{2}-\d{4}|\d{9})',
#         'balance': r'\b(?:balance|amount|sum|total)(?:\s+(?:due|owed|outstanding|unpaid))?\s*(?:is|of|:)?\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
#         'date_of_birth': r'\b(?:DOB|date of birth|birth date|born on)\s*(?:is|:|-)?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
#         'address': r'\b(?:address|residence|live at|living at|located at)\s*(?:is|:|-)?\s*(\d+\s+[^\d]+(?:avenue|ave|street|st|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|circle|cir|trail|trl|way|place|pl)\b.{5,50})',
#     }
#     return patterns
#
#
# def load_verification_patterns():
#     """Load patterns for identity verification detection"""
#     patterns = {
#         'verification_question': r'\b(?:verify|confirm|validation|identify|authentication|prove)(?:\s+(?:your|identity|yourself|who you are))?\b',
#         'dob_verification': r'\b(?:DOB|date of birth|birthday|born on|birth date)\b',
#         'ssn_verification': r'\b(?:SSN|social security|last four|last 4)\b',
#         'address_verification': r'\b(?:address|zip code|postal code|residence)\b',
#         'verification_success': r'\b(?:thank you for verifying|verification successful|identity confirmed|that matches|that\'s correct|verification complete)\b'
#     }
#     return patterns
#
#
# def detect_sensitive_information(text, patterns):
#     """Detect sensitive information in text using regex patterns"""
#     if not text or not isinstance(text, str):
#         return {}, []
#
#     text = text.lower()
#     found_info = {}
#     information_types = []
#
#     for info_type, pattern in patterns.items():
#         matches = re.findall(pattern, text)
#         if matches:
#             found_info[info_type] = matches
#             information_types.append(info_type)
#
#     return found_info, information_types
#
#
# def detect_verification(text, patterns):
#     """Detect identity verification in text using regex patterns"""
#     if not text or not isinstance(text, str):
#         return False, []
#
#     text = text.lower()
#     verification_types = []
#
#     for verification_type, pattern in patterns.items():
#         if re.search(pattern, text):
#             verification_types.append(verification_type)
#
#     # Consider verification present if we find any verification pattern
#     return len(verification_types) > 0, verification_types
#
#
# def analyze_compliance_pattern_matching(conversations):
#     """
#     Analyze conversations for compliance violations using pattern matching
#     Returns calls where sensitive info was shared without proper verification
#     """
#     sensitive_patterns = load_sensitive_info_patterns()
#     verification_patterns = load_verification_patterns()
#
#     results = {
#         'violations': defaultdict(list),
#         'all_sensitive_info': defaultdict(list)
#     }
#
#     for call_id, conversation in conversations.items():
#         # Track verification status throughout the call
#         verification_done = False
#         verification_types = []
#
#         # Track sensitive information shared
#         sensitive_info_shared = []
#
#         # Process the conversation chronologically
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             # Check for verification (can be done by agent or confirmed by borrower)
#             is_verification, found_verification = detect_verification(text, verification_patterns)
#             if is_verification:
#                 verification_done = True
#                 verification_types.extend(found_verification)
#
#             # Check for sensitive information in agent utterances
#             if 'agent' in speaker:
#                 sensitive_info, info_types = detect_sensitive_information(text, sensitive_patterns)
#
#                 if sensitive_info:
#                     # Record all instances of sensitive information
#                     results['all_sensitive_info'][call_id].append({
#                         'text': text,
#                         'sensitive_info': sensitive_info,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#                     # If sensitive info shared before verification, record violation
#                     if not verification_done and sensitive_info:
#                         results['violations'][call_id].append({
#                             'text': text,
#                             'sensitive_info': sensitive_info,
#                             'verification_status': 'Not verified',
#                             'stime': utterance.get('stime'),
#                             'etime': utterance.get('etime')
#                         })
#
#     return results
#
#
# # =====================
# # Compliance Detection - ML-based Approach
# # =====================
#
# class ComplianceMLDetector:
#     """ML-based detector for compliance violations"""
#
#     def __init__(self):
#         """Initialize the ML-based compliance detector"""
#         # In a real implementation, you would load a trained model here
#         # This is a simplified version for demonstration
#         self.sensitive_patterns = load_sensitive_info_patterns()
#         self.verification_patterns = load_verification_patterns()
#
#         # Additional contextual phrases for better detection
#         self.sensitive_info_indicators = [
#             "your account", "your balance", "you owe", "outstanding amount",
#             "personal information", "payment details", "account status"
#         ]
#
#         self.verification_indicators = [
#             "can you confirm", "need to verify", "for security purposes",
#             "before we proceed", "to protect your privacy", "authenticate"
#         ]
#
#     def detect_sensitive_information(self, text):
#         """ML-enhanced detection of sensitive information"""
#         # First use regex patterns
#         found_info, info_types = detect_sensitive_information(text, self.sensitive_patterns)
#
#         # Then enhance with contextual understanding
#         text_lower = text.lower()
#         for indicator in self.sensitive_info_indicators:
#             if indicator in text_lower and not info_types:
#                 # Mark as potentially containing implicit sensitive information
#                 info_types.append("implicit_sensitive_info")
#                 found_info["implicit_sensitive_info"] = ["contextual reference"]
#                 break
#
#         return found_info, info_types
#
#     def detect_verification(self, text):
#         """ML-enhanced detection of verification"""
#         # First use regex patterns
#         is_verification, verification_types = detect_verification(text, self.verification_patterns)
#
#         # Then enhance with contextual understanding
#         if not is_verification:
#             text_lower = text.lower()
#             for indicator in self.verification_indicators:
#                 if indicator in text_lower:
#                     verification_types.append("implicit_verification")
#                     is_verification = True
#                     break
#
#         return is_verification, verification_types
#
#
# def analyze_compliance_ml(conversations):
#     """
#     Analyze conversations for compliance violations using ML approach
#     Returns calls where sensitive info was shared without proper verification
#     """
#     detector = ComplianceMLDetector()
#
#     results = {
#         'violations': defaultdict(list),
#         'all_sensitive_info': defaultdict(list)
#     }
#
#     for call_id, conversation in conversations.items():
#         # Track verification status throughout the call
#         verification_done = False
#         verification_types = []
#
#         # Track sensitive information shared
#         sensitive_info_shared = []
#
#         # Process the conversation chronologically
#         for utterance in conversation:
#             speaker = utterance.get('speaker', '').lower()
#             text = utterance.get('text', '')
#
#             # Check for verification (can be done by agent or confirmed by borrower)
#             is_verification, found_verification = detector.detect_verification(text)
#             if is_verification:
#                 verification_done = True
#                 verification_types.extend(found_verification)
#
#             # Check for sensitive information in agent utterances
#             if 'agent' in speaker:
#                 sensitive_info, info_types = detector.detect_sensitive_information(text)
#
#                 if sensitive_info:
#                     # Record all instances of sensitive information
#                     results['all_sensitive_info'][call_id].append({
#                         'text': text,
#                         'sensitive_info': sensitive_info,
#                         'stime': utterance.get('stime'),
#                         'etime': utterance.get('etime')
#                     })
#
#                     # If sensitive info shared before verification, record violation
#                     if not verification_done and sensitive_info:
#                         results['violations'][call_id].append({
#                             'text': text,
#                             'sensitive_info': sensitive_info,
#                             'verification_status': 'Not verified',
#                             'stime': utterance.get('stime'),
#                             'etime': utterance.get('etime')
#                         })
#
#     return results
#
#
# # =====================
# # Compliance Detection - LLM-based Approach
# # =====================
#
# def analyze_compliance_llm(conversations):
#     """
#     Simulate LLM-based compliance violation detection
#     Note: In a real implementation, this would call an actual LLM API
#     """
#     # For demonstration purposes, we'll use our pattern matching implementation
#     # In a real-world scenario, this would call an LLM API with appropriate prompting
#     return analyze_compliance_pattern_matching(conversations)
#
#
# # =====================
# # Data Loading Functions
# # =====================
#
# def load_data(file_content):
#     """Load conversation data from a JSON string or file"""
#     try:
#         if isinstance(file_content, str):
#             # Try to parse as JSON
#             return json.loads(file_content)
#         else:
#             # Read the file and parse as JSON
#             data = json.load(file_content)
#             return data
#     except json.JSONDecodeError as e:
#         st.error(f"Error parsing JSON: {e}")
#         return None
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return None
#
#
# # =====================
# # Visualization Functions
# # =====================
#
# def create_profanity_chart(results):
#     """Create a chart showing profanity statistics"""
#     # Prepare data for visualization
#     agent_counts = {}
#     borrower_counts = {}
#
#     for call_id, instances in results['agent_profanity'].items():
#         agent_counts[call_id] = len(instances)
#
#     for call_id, instances in results['borrower_profanity'].items():
#         borrower_counts[call_id] = len(instances)
#
#     # Convert to DataFrame
#     all_call_ids = set(list(agent_counts.keys()) + list(borrower_counts.keys()))
#     data = []
#
#     for call_id in all_call_ids:
#         data.append({
#             'Call ID': call_id,
#             'Agent Profanity Instances': agent_counts.get(call_id, 0),
#             'Borrower Profanity Instances': borrower_counts.get(call_id, 0)
#         })
#
#     df = pd.DataFrame(data)
#
#     # Create chart
#     fig, ax = plt.subplots(figsize=(10, 6))
#     df.plot(x='Call ID', kind='bar', stacked=False, ax=ax)
#     plt.title('Profanity Instances by Call')
#     plt.ylabel('Number of Instances')
#     plt.tight_layout()
#
#     return fig
#
#
# def create_compliance_chart(results):
#     """Create a chart showing compliance violation statistics"""
#     # Prepare data for visualization
#     violation_counts = {}
#     sensitive_info_counts = {}
#
#     for call_id, instances in results['violations'].items():
#         violation_counts[call_id] = len(instances)
#
#     for call_id, instances in results['all_sensitive_info'].items():
#         sensitive_info_counts[call_id] = len(instances)
#
#     # Convert to DataFrame
#     all_call_ids = set(list(violation_counts.keys()) + list(sensitive_info_counts.keys()))
#     data = []
#
#     for call_id in all_call_ids:
#         data.append({
#             'Call ID': call_id,
#             'Compliance Violations': violation_counts.get(call_id, 0),
#             'Sensitive Info Instances': sensitive_info_counts.get(call_id, 0)
#         })
#
#     df = pd.DataFrame(data)
#
#     # Create chart
#     fig, ax = plt.subplots(figsize=(10, 6))
#     df.plot(x='Call ID', kind='bar', stacked=False, ax=ax)
#     plt.title('Compliance Statistics by Call')
#     plt.ylabel('Number of Instances')
#     plt.tight_layout()
#
#     return fig
#
#
# # =====================
# # Streamlit App
# # =====================
#
# def run_call_analysis_app():
#     st.title("Debt Collection Call Analysis")
#
#     # File upload
#     uploaded_file = st.file_uploader("Upload a conversation file (JSON)", type=["json"])
#
#     # Two columns for configuration
#     col1, col2 = st.columns(2)
#
#     with col1:
#         # Select entity type
#         entity_type = st.selectbox(
#             "Select Analysis Type",
#             ["Profanity Detection", "Privacy and Compliance Violation"]
#         )
#
#     with col2:
#         # Select approach
#         approach = st.selectbox(
#             "Select Analysis Approach",
#             ["Pattern Matching", "Machine Learning", "LLM-based"]
#         )
#
#     if uploaded_file is not None:
#         # Load the data
#         conversation_data = load_data(uploaded_file)
#
#         if conversation_data:
#             # For the sample format, wrap in a dictionary with a generated call ID
#             call_id = uploaded_file.name.split('.')[0]
#             conversations = {call_id: conversation_data}
#
#             if st.button("Analyze"):
#                 st.subheader("Analysis Results")
#
#                 if entity_type == "Profanity Detection":
#                     with st.spinner("Analyzing for profanity..."):
#                         if approach == "Pattern Matching":
#                             results = analyze_profanity_pattern_matching(conversations)
#                         elif approach == "Machine Learning":
#                             results = analyze_profanity_ml(conversations)
#                         else:  # LLM-based
#                             results = analyze_profanity_llm(conversations)
#
#                         agent_profanity = results['agent_profanity']
#                         borrower_profanity = results['borrower_profanity']
#
#                         # Display visualization
#                         if agent_profanity or borrower_profanity:
#                             st.subheader("Profanity Statistics")
#                             fig = create_profanity_chart(results)
#                             st.pyplot(fig)
#
#                         # Display results
#                         if not agent_profanity and not borrower_profanity:
#                             st.success("No profanity detected in the conversation.")
#                         else:
#                             # Display agent profanity
#                             if agent_profanity:
#                                 st.error("Agent Profanity Detected:")
#                                 for call_id, instances in agent_profanity.items():
#                                     st.write(f"**Call ID: {call_id}**")
#                                     for i, instance in enumerate(instances):
#                                         st.write(f"Instance {i + 1}:")
#                                         st.write(f"- Text: *{instance['text']}*")
#                                         st.write(f"- Profane words: {', '.join(instance['profane_words'])}")
#                                         st.write(f"- Time: {instance['stime']} - {instance['etime']}")
#                                         st.write("---")
#                             else:
#                                 st.success("No profanity detected from the agent.")
#
#                             # Display borrower profanity
#                             if borrower_profanity:
#                                 st.warning("Customer/Borrower Profanity Detected:")
#                                 for call_id, instances in borrower_profanity.items():
#                                     st.write(f"**Call ID: {call_id}**")
#                                     for i, instance in enumerate(instances):
#                                         st.write(f"Instance {i + 1}:")
#                                         st.write(f"- Text: *{instance['text']}*")
#                                         st.write(f"- Profane words: {', '.join(instance['profane_words'])}")
#                                         st.write(f"- Time: {instance['stime']} - {instance['etime']}")
#                                         st.write("---")
#                             else:
#                                 st.success("No profanity detected from the customer/borrower.")
#                 else:  # Privacy and Compliance Violation
#                     with st.spinner("Analyzing for compliance violations..."):
#                         if approach == "Pattern Matching":
#                             results = analyze_compliance_pattern_matching(conversations)
#                         elif approach == "Machine Learning":
#                             results = analyze_compliance_ml(conversations)
#                         else:  # LLM-based
#                             results = analyze_compliance_llm(conversations)
#
#                         violations = results['violations']
#                         all_sensitive_info = results['all_sensitive_info']
#
#                         # Display visualization
#                         if violations or all_sensitive_info:
#                             st.subheader("Compliance Violation Statistics")
#                             fig = create_compliance_chart(results)
#                             st.pyplot(fig)
#
#                         # Display results
#                         if not violations:
#                             st.success("No compliance violations detected.")
#                         else:
#                             st.error("Compliance Violations Detected:")
#                             for call_id, instances in violations.items():
#                                 st.write(f"**Call ID: {call_id}**")
#                                 for i, instance in enumerate(instances):
#                                     st.write(f"Violation {i + 1}:")
#                                     st.write(f"- Text: *{instance['text']}*")
#                                     st.write(f"- Sensitive info shared: {', '.join(instance['sensitive_info'].keys())}")
#                                     st.write(f"- Verification status: {instance['verification_status']}")
#                                     st.write(f"- Time: {instance['stime']} - {instance['etime']}")
#                                     st.write("---")
#
#                         # Display all sensitive information for reference
#                         st.subheader("All Sensitive Information")
#                         if not all_sensitive_info:
#                             st.info("No sensitive information detected.")
#                         else:
#                             for call_id, instances in all_sensitive_info.items():
#                                 st.write(f"**Call ID: {call_id}**")
#                                 for i, instance in enumerate(instances):
#                                     st.write(f"Instance {i + 1}:")
#                                     st.write(f"- Text: *{instance['text']}*")
#                                     st.write(f"- Info types: {', '.join(instance['sensitive_info'].keys())}")
#                                     st.write(f"- Time: {instance['stime']} - {instance['etime']}")
#                                     st.write("---")
#         else:
#             st.error("Failed to load the conversation data. Please check the file format.")
#
#
# if __name__ == "__main__":
#     run_call_analysis_app()