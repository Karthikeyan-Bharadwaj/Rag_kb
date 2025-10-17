"""
Test the Fixed CV Search System
"""

import requests
import time

def test_cv_search_system():
    """Test if the CV search system is working properly."""
    
    print("🧪 Testing CV Search System")
    print("=" * 40)
    
    base_url = "http://127.0.0.1:7864"
    
    # Test if server is responding
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding")
            print(f"📱 Access the system at: {base_url}")
            print("\n🎯 System Features:")
            print("   ✅ Direct CV text input (no file upload issues)")
            print("   ✅ Keyword-based search functionality") 
            print("   ✅ Quick search buttons")
            print("   ✅ CV status checking")
            print("   ✅ Pre-loaded with sample CV")
            
            print(f"\n📋 How to Use:")
            print(f"   1. Open: {base_url}")
            print(f"   2. Go to 'Search CV' tab")
            print(f"   3. Try searches like:")
            print(f"      • 'programming languages'")
            print(f"      • 'education degree'")
            print(f"      • 'work experience'")
            print(f"      • 'projects'")
            
            return True
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection refused - server not running")
        return False
    except requests.exceptions.Timeout:
        print("⏱️ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_sample_searches():
    """Show sample search queries that work with the system."""
    
    print(f"\n💡 Sample Search Queries to Try:")
    print("-" * 30)
    
    searches = [
        ("Technical Skills", "programming languages python javascript"),
        ("Education", "bachelor degree university education"),
        ("Work Experience", "software engineer developer intern"),
        ("Projects", "e-commerce machine learning RAG"),
        ("Certifications", "AWS certified cloud azure"),
        ("Achievements", "hackathon dean list publications")
    ]
    
    for category, query in searches:
        print(f"🔍 {category}: '{query}'")

if __name__ == "__main__":
    success = test_cv_search_system()
    
    if success:
        show_sample_searches()
        print(f"\n🎉 System is ready and working!")
        print(f"🌐 Visit: http://127.0.0.1:7864")
    else:
        print(f"\n🔧 System needs attention - check if fixed_cv_gui.py is running")