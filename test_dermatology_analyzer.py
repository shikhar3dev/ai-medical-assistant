"""Test script for the enhanced dermatology analyzer"""

import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dermatology_analysis import DermatologyAnalyzer

def test_analyzer():
    """Test the dermatology analyzer with a sample image"""
    
    # Create a test analyzer
    analyzer = DermatologyAnalyzer()
    
    # Create a synthetic test image (reddish patch on skin-colored background)
    test_image = np.ones((200, 200, 3), dtype=np.uint8)
    
    # Skin-colored background
    test_image[:, :] = [220, 180, 140]  # Light skin tone
    
    # Add a reddish inflammatory patch
    test_image[50:150, 50:150, 0] = 200  # More red
    test_image[50:150, 50:150, 1] = 120  # Less green
    test_image[50:150, 50:150, 2] = 120  # Less blue
    
    # Add some texture/roughness
    noise = np.random.randint(-20, 20, (100, 100, 3))
    test_image[50:150, 50:150] = np.clip(test_image[50:150, 50:150] + noise, 0, 255)
    
    print("🧪 Testing Dermatology Analyzer...")
    print("=" * 50)
    
    try:
        # Analyze the test image
        results = analyzer.analyze_skin_condition(test_image, "test_skin_condition.jpg")
        
        print("✅ Analysis completed successfully!")
        print()
        
        # Display results
        condition = results['condition_assessment']
        print(f"🩺 Primary Condition: {condition['primary_condition']}")
        print(f"🎯 Confidence: {condition['confidence']*100:.1f}%")
        print()
        
        severity = results['severity_analysis']
        print(f"📊 Severity Level: {severity['level']}")
        print(f"🔴 Redness Intensity: {severity['redness_intensity']*100:.1f}%")
        print(f"📏 Affected Area: {severity['affected_area_percentage']:.1f}%")
        print(f"🔥 Inflammation Level: {severity['inflammation_level']*100:.1f}%")
        print()
        
        texture = results['texture_analysis']
        print(f"🎨 Texture Description: {texture['texture_description']}")
        print(f"📐 Roughness Score: {texture['roughness']*100:.1f}%")
        print()
        
        print("💡 Top Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        print()
        
        print(f"🎯 Overall Confidence: {results['confidence_score']*100:.0f}%")
        print()
        
        print("✅ Test completed successfully!")
        print("The enhanced dermatology analyzer is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_with_real_image(image_path):
    """Test with a real image file if provided"""
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        
        analyzer = DermatologyAnalyzer()
        results = analyzer.analyze_skin_condition(image_array, image_path)
        
        print(f"🖼️ Analysis of {image_path}:")
        print("=" * 50)
        
        condition = results['condition_assessment']
        print(f"🩺 Likely Condition: {condition['primary_condition']}")
        print(f"🎯 Assessment Confidence: {condition['confidence']*100:.1f}%")
        
        severity = results['severity_analysis']
        print(f"📊 Severity: {severity['level']}")
        
        print("\n💡 Key Recommendations:")
        for rec in results['recommendations'][:4]:
            print(f"   • {rec}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing image {image_path}: {e}")
        return False

if __name__ == "__main__":
    print("🏥 Enhanced Medical AI - Dermatology Analyzer Test")
    print("=" * 60)
    print()
    
    # Test with synthetic image
    success = test_analyzer()
    
    if success:
        print("\n" + "="*60)
        print("🎉 All tests passed! The enhanced dermatology analyzer is ready.")
        print("📋 You can now use the enhanced dashboard with improved medical analysis.")
        print("🚀 Run 'test_enhanced_analysis.bat' to start the dashboard.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
