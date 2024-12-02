import pytest
from src import config
import os

def test_model_options_structure():
    """Test the structure and content of MODEL_OPTIONS"""
    assert isinstance(config.MODEL_OPTIONS, dict)
    
    # Test required providers exist
    required_providers = ['ollama', 'openai', 'azure_openai', 'anthropic', 'google']
    for provider in required_providers:
        assert provider in config.MODEL_OPTIONS
        assert isinstance(config.MODEL_OPTIONS[provider], list)
        assert len(config.MODEL_OPTIONS[provider]) > 0

def test_pricing_structure():
    """Test the structure and content of PRICING"""
    assert isinstance(config.PRICING, dict)
    
    # Test required providers exist
    required_providers = ['azure_openai', 'openai', 'anthropic', 'google']
    for provider in required_providers:
        assert provider in config.PRICING
        assert isinstance(config.PRICING[provider], dict)
        
        # Test each provider has at least one model with pricing
        assert len(config.PRICING[provider]) > 0
        
        # Test pricing values are positive floats
        for model, price in config.PRICING[provider].items():
            assert isinstance(price, float)
            assert price > 0

def test_max_retries():
    """Test MAX_RETRIES configuration"""
    assert isinstance(config.MAX_RETRIES, int)
    assert config.MAX_RETRIES > 0

def test_default_temperatures():
    """Test default temperature settings"""
    assert hasattr(config, 'EVAL_TEMPERATURE')
    assert hasattr(config, 'OPTIM_TEMPERATURE')
    assert isinstance(config.EVAL_TEMPERATURE, float)
    assert isinstance(config.OPTIM_TEMPERATURE, float)
    assert 0 <= config.EVAL_TEMPERATURE <= 1
    assert 0 <= config.OPTIM_TEMPERATURE <= 1

def test_model_provider_consistency():
    """Test consistency between MODEL_OPTIONS and PRICING"""
    # All models in PRICING should exist in MODEL_OPTIONS
    for provider in config.PRICING:
        if provider == 'ollama':  # Skip ollama as it's free/local
            continue
        assert provider in config.MODEL_OPTIONS
        for model in config.PRICING[provider]:
            # Check if the model or a version of it exists in MODEL_OPTIONS
            model_exists = any(model in m for m in config.MODEL_OPTIONS[provider])
            assert model_exists, f"Model {model} not found in MODEL_OPTIONS for {provider}"

def test_global_config_variables():
    """Test global configuration variables"""
    global_vars = [
        'SELECTED_PROVIDER',
        'MODEL_NAME',
        'EVAL_PROVIDER',
        'EVAL_MODEL',
        'OPTIM_PROVIDER',
        'OPTIM_MODEL'
    ]
    
    for var in global_vars:
        assert hasattr(config, var)
        # Initially these should be None
        assert getattr(config, var) is None

def test_model_compatibility():
    """Test model compatibility across providers"""
    for provider, models in config.MODEL_OPTIONS.items():
        # Skip ollama as it's free/local
        if provider != 'ollama' and provider in config.PRICING:
            # Each model should have a corresponding price
            for model in models:
                price_exists = any(model in m for m in config.PRICING[provider])
                assert price_exists, f"No price found for model {model} in {provider}"

def test_pricing_ranges():
    """Test that pricing falls within expected ranges"""
    for provider, models in config.PRICING.items():
        for model, price in models.items():
            # Prices should be reasonable (less than $1 per 1000 tokens)
            assert price < 1.0, f"Price for {model} in {provider} seems unusually high"
            # Prices should be positive
            assert price > 0, f"Price for {model} in {provider} should be positive"

def test_temperature_ranges():
    """Test temperature configurations are within valid ranges"""
    temperatures = [
        config.EVAL_TEMPERATURE,
        config.OPTIM_TEMPERATURE
    ]
    
    for temp in temperatures:
        assert isinstance(temp, float)
        assert 0 <= temp <= 1, "Temperature should be between 0 and 1"

def test_model_naming_conventions():
    """Test model naming conventions are consistent"""
    for provider, models in config.MODEL_OPTIONS.items():
        for model in models:
            # Model names should be lowercase and use hyphens or dots
            assert model.islower(), f"Model name {model} should be lowercase"
            assert all(c.isalnum() or c in '-.' for c in model), \
                f"Model name {model} contains invalid characters" 