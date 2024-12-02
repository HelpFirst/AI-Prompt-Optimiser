import pytest
from src import model_interface
from unittest.mock import patch, MagicMock
import os
import json
from pathlib import Path

@pytest.fixture
def mock_response():
    return {
        'choices': [{
            'message': {
                'content': 'Test response'
            }
        }]
    }

@pytest.fixture
def setup_env():
    """Setup environment variables for testing"""
    os.environ['OPENAI_API_KEY'] = 'test_key'
    os.environ['ANTHROPIC_API_KEY'] = 'test_key'
    os.environ['GOOGLE_API_KEY'] = 'test_key'
    yield
    del os.environ['OPENAI_API_KEY']
    del os.environ['ANTHROPIC_API_KEY']
    del os.environ['GOOGLE_API_KEY']

# Tests for model output with different providers
def test_get_model_output_cache(tmp_path):
    """Test that caching works correctly in get_model_output"""
    with patch('src.model_interface.CACHE_DIR', tmp_path):
        with patch('src.model_interface._get_openai_output') as mock_openai:
            mock_openai.side_effect = Exception("API Error")
            
            provider = "openai"
            model = "gpt-3.5-turbo"
            temperature = 0.7
            full_prompt = "Test prompt"
            text = "Test input"
            
            with pytest.raises(RuntimeError):
                model_interface.get_model_output(
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    full_prompt=full_prompt,
                    text=text,
                    use_cache=True
                )
            
            assert mock_openai.call_count == model_interface.MAX_RETRIES

def test_get_model_output_no_cache(setup_env):
    """Test get_model_output with caching disabled"""
    with patch('src.model_interface._get_openai_output') as mock_openai:
        mock_openai.side_effect = Exception("API Error")
        
        provider = "openai"
        model = "gpt-3.5-turbo"
        temperature = 0.7
        full_prompt = "Test prompt"
        text = "Test input"
        
        with pytest.raises(RuntimeError):
            model_interface.get_model_output(
                provider=provider,
                model=model,
                temperature=temperature,
                full_prompt=full_prompt,
                text=text,
                use_cache=False
            )

@patch('src.model_interface._get_openai_output')
def test_get_model_output_openai(mock_openai_output, setup_env, mock_response):
    """Test OpenAI integration"""
    mock_openai_output.return_value = mock_response
    
    result = model_interface.get_model_output(
        provider="openai",
        model="gpt-3.5-turbo",
        temperature=0.7,
        full_prompt="Test prompt",
        text="Test input",
        use_cache=False
    )
    
    assert isinstance(result, dict)
    assert 'choices' in result
    assert isinstance(result['choices'], list)
    assert len(result['choices']) > 0
    assert 'message' in result['choices'][0]
    assert 'content' in result['choices'][0]['message']
    
    mock_openai_output.assert_called_once_with(
        "gpt-3.5-turbo",
        "Test prompt",
        "Test input",
        False,  # use_json_mode
        0.7     # temperature
    )

@patch('anthropic.Anthropic')
def test_get_model_output_anthropic(mock_anthropic, setup_env, mock_response):
    """Test Anthropic integration"""
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_client.completions.create.return_value = MagicMock(completion="Test response")
    
    result = model_interface.get_model_output(
        provider="anthropic",
        model="claude-2",
        temperature=0.7,
        full_prompt="Test prompt",
        text="Test input",
        use_cache=False
    )
    
    assert isinstance(result, dict)
    assert 'choices' in result
    assert result['choices'][0]['message']['content'] == "Test response"

@patch('google.generativeai.GenerativeModel')
def test_get_model_output_google(mock_genai_model, setup_env, mock_response):
    """Test Google AI integration"""
    mock_model = MagicMock()
    mock_genai_model.return_value = mock_model
    mock_model.generate_content.return_value = MagicMock(text="Test response")
    
    result = model_interface.get_model_output(
        provider="google",
        model="gemini-pro",
        temperature=0.7,
        full_prompt="Test prompt",
        text="Test input",
        use_cache=False
    )
    
    assert isinstance(result, dict)
    assert 'choices' in result
    assert result['choices'][0]['message']['content'] == "Test response"

@patch('ollama.chat')
def test_get_model_output_ollama(mock_ollama_chat, mock_response):
    """Test Ollama integration"""
    mock_ollama_chat.return_value = {'message': {'content': 'Test response'}}
    
    result = model_interface.get_model_output(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        full_prompt="Test prompt",
        text="Test input",
        use_cache=False
    )
    
    assert isinstance(result, dict)
    assert 'choices' in result
    assert result['choices'][0]['message']['content'] == "Test response"

# Tests for analysis functions
@patch('src.model_interface._get_openai_analysis')
def test_get_analysis_openai(mock_openai_analysis, setup_env):
    """Test analysis generation with OpenAI"""
    mock_openai_analysis.return_value = "Test analysis"
    
    result = model_interface.get_analysis(
        provider="openai",
        model="gpt-3.5-turbo",
        temperature=0.7,
        analysis_prompt="Analyze this"
    )
    
    assert isinstance(result, str)
    assert result == "Test analysis"
    mock_openai_analysis.assert_called_once_with(
        "gpt-3.5-turbo",
        "Analyze this",
        0.7
    )

# Tests for caching functionality
def test_cache_operations(tmp_path):
    """Test cache key generation and cache operations"""
    with patch('src.model_interface.CACHE_DIR', tmp_path):
        # Test cache key generation
        cache_key = model_interface.get_cache_key(
            full_prompt="Test prompt",
            text="Test input",
            model="gpt-3.5-turbo"
        )
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length
        
        # Test cache output
        test_output = {'test': 'data'}
        model_interface.cache_output(cache_key, test_output)
        cache_file = tmp_path / f"{cache_key}.json"
        assert cache_file.exists()
        
        # Test cache retrieval
        cached_output = model_interface.get_cached_output(cache_key)
        assert cached_output == test_output
        
        # Test non-existent cache
        non_existent = model_interface.get_cached_output("nonexistent")
        assert non_existent is None

def test_invalid_provider():
    """Test handling of invalid provider"""
    with pytest.raises(RuntimeError, match="Failed to get model output after .* retries"):
        model_interface.get_model_output(
            provider="invalid_provider",
            model="test_model",
            temperature=0.7,
            full_prompt="Test prompt",
            text="Test input"
        )