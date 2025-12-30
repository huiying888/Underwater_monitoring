from detection_backend import DetectionBackend
from gui_frontend import GUIFrontend



def main():
    # Initialize backend
    backend = DetectionBackend()
    
    # Initialize frontend with backend
    frontend = GUIFrontend(backend)
    
    # Run the application
    frontend.run()


if __name__ == "__main__":
    main()