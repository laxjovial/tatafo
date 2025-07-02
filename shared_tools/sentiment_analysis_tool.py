# shared_tools/sentiment_analysis_tool.py

import logging
from typing import Dict, Any
from langchain_core.tools import tool

# For a real implementation, you might use:
# from transformers import pipeline # For local models
# from textblob import TextBlob # For simple rule-based
# Or integrate with an external sentiment analysis API

logger = logging.getLogger(__name__)

@tool
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes the sentiment of a given text and returns a classification (positive, neutral, negative)
    and a score. This tool is useful for understanding the emotional tone of written content.

    Args:
        text (str): The input text to analyze sentiment for.

    Returns:
        Dict[str, Any]: A dictionary containing 'sentiment' (str) and 'score' (float).
                        Sentiment can be 'positive', 'neutral', or 'negative'.
                        Score is typically between -1.0 (most negative) and 1.0 (most positive).
                        Returns an error message if analysis fails.
    """
    logger.info(f"Tool: analyze_sentiment called for text (first 50 chars): '{text[:50]}...'")

    if not text or not isinstance(text, str):
        return {"error": "Input text must be a non-empty string."}

    try:
        # --- Mock Sentiment Analysis ---
        # In a real application, you would integrate with a sentiment analysis library or API here.
        # Examples:
        # 1. Using TextBlob (simple, rule-based):
        #    from textblob import TextBlob
        #    analysis = TextBlob(text)
        #    polarity = analysis.sentiment.polarity
        #    if polarity > 0.1:
        #        sentiment = "positive"
        #    elif polarity < -0.1:
        #        sentiment = "negative"
        #    else:
        #        sentiment = "neutral"
        #    score = polarity
        
        # 2. Using Hugging Face Transformers (requires model download):
        #    sentiment_pipeline = pipeline("sentiment-analysis")
        #    result = sentiment_pipeline(text)[0]
        #    sentiment = result['label'].lower() # e.g., 'positive', 'negative'
        #    score = result['score'] if sentiment == 'positive' else -result['score'] # Adjust score for negative

        # For now, a simple keyword-based mock:
        text_lower = text.lower()
        if "excellent" in text_lower or "great" in text_lower or "happy" in text_lower or "positive" in text_lower:
            sentiment = "positive"
            score = 0.8
        elif "bad" in text_lower or "terrible" in text_lower or "sad" in text_lower or "negative" in text_lower:
            sentiment = "negative"
            score = -0.7
        else:
            sentiment = "neutral"
            score = 0.0

        logger.info(f"Sentiment analysis result: Sentiment='{sentiment}', Score={score}")
        return {"sentiment": sentiment, "score": score}

    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
        return {"error": f"Failed to analyze sentiment: {e}"}

# CLI Test (optional)
if __name__ == "__main__":
    print("--- Testing analyze_sentiment tool ---")

    test_texts = [
        "This is an excellent product, I'm very happy with it!",
        "The service was terrible, I'm quite disappointed.",
        "The weather today is neither good nor bad.",
        "I have a neutral opinion on this matter.",
        "This is a fantastic opportunity.",
        "I am so angry right now."
    ]

    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text}'")
        result = analyze_sentiment(text)
        print(f"Result: {result}")
        assert "sentiment" in result and "score" in result, "Sentiment analysis result missing keys."
        if "excellent" in text.lower() or "happy" in text.lower() or "fantastic" in text.lower():
            assert result["sentiment"] == "positive"
        elif "terrible" in text.lower() or "disappointed" in text.lower() or "angry" in text.lower():
            assert result["sentiment"] == "negative"
        else:
            assert result["sentiment"] == "neutral"
    
    print("\n--- Testing with invalid input ---")
    invalid_result = analyze_sentiment(123) # type: ignore
    print(f"Result for invalid input: {invalid_result}")
    assert "error" in invalid_result, "Expected error for invalid input."

    empty_result = analyze_sentiment("")
    print(f"Result for empty input: {empty_result}")
    assert "error" in empty_result, "Expected error for empty input."

    print("\nAll sentiment analysis tests passed (mocked).")
