"""Main application entry point for deployment."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the enhanced dashboard
if __name__ == "__main__":
    # Set environment variables for deployment
    os.environ['STREAMLIT_SERVER_PORT'] = os.environ.get('PORT', '8501')
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    
    # Import and run the dashboard
    from enhanced_dashboard import main
    main()
