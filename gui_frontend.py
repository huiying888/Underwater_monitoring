import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque


class GUIFrontend:
    def __init__(self, backend):
        self.backend = backend
        self.setup_gui()
        self.image_item = None
        self.detection_duration = 0  # 0 = unlimited, >0 = seconds
        self.detection_start_time = None
        self.timer_job = None
        # Graph data
        self.detection_times = deque(maxlen=50)
        self.detection_counts = deque(maxlen=50)
        self.fish_counts = deque(maxlen=50)  # For fish count tracking
        self.graph_update_job = None
        
    def setup_gui(self):
        """Initialize the GUI components"""
        self.root = tk.Tk()
        self.root.title("🐠 Underwater Detection System")
        self.root.state("zoomed")
        self.root.configure(bg="#1e3a5f")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TCombobox', fieldbackground='#ffffff', background='#4a90e2')
        
        # Main container with gradient-like effect
        main_frame = tk.Frame(self.root, bg="#1e3a5f")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side: video with full space
        video_frame = tk.Frame(main_frame, bg="#2c5282", relief="raised", bd=2)
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        video_title = tk.Label(video_frame, text="📹 Live Detection Feed", 
                              font=("Segoe UI", 14, "bold"), bg="#2c5282", fg="white")
        video_title.pack(pady=10)
        
        self.canvas = tk.Canvas(video_frame, bg="#1a202c", highlightthickness=0, relief="sunken", bd=2)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Right side: modern control panel (wider for more buttons)
        control_frame = tk.Frame(main_frame, bg="#2d3748", width=650, relief="raised", bd=2)
        control_frame.pack(side="right", fill="y")
        control_frame.pack_propagate(False)
        
        # Header with gradient effect
        header_frame = tk.Frame(control_frame, bg="#4a90e2", height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        # Exit button in top right corner
        tk.Button(header_frame, text="✕", bg="#e53e3e", fg="white",
                 activebackground="#c53030", font=("Segoe UI", 12, "bold"),
                 relief="flat", bd=0, pady=5, padx=8, cursor="hand2",
                 command=self.root.destroy).pack(side="right", padx=10, pady=10)
        
        title_label = tk.Label(header_frame, text="🎯 Detection Controls", 
                              font=("Segoe UI", 20, "bold"), bg="#4a90e2", fg="white")
        title_label.pack(expand=True)
        
        # Content area - split into top and bottom halves
        content_frame = tk.Frame(control_frame, bg="#2d3748")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # TOP HALF - split left and right
        top_frame = tk.Frame(content_frame, bg="#2d3748")
        top_frame.pack(fill="x", pady=(0, 10))
        
        # TOP LEFT - controls
        top_left = tk.Frame(top_frame, bg="#2d3748")
        top_left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Detection type dropdown
        type_label = tk.Label(top_left, text="Detection Type:", 
                             font=("Segoe UI", 11, "bold"), bg="#2d3748", fg="#e2e8f0")
        type_label.pack(anchor="w", pady=(0, 3))
        
        self.detection_var = tk.StringVar(value="Asset Detection")
        self.detection_combo = ttk.Combobox(top_left, textvariable=self.detection_var,
                                           values=["Asset Detection", "Fish Detection"],
                                           state="readonly", font=("Segoe UI", 10),
                                           style='Custom.TCombobox')
        self.detection_combo.pack(fill="x", pady=(0, 8))
        self.detection_combo.bind('<<ComboboxSelected>>', self.on_detection_change)
        
        # Status
        status_frame = tk.Frame(top_left, bg="#4a5568", relief="ridge", bd=1)
        status_frame.pack(fill="x", pady=(0, 8))
        
        tk.Label(status_frame, text="📊 Status:", font=("Segoe UI", 9, "bold"), 
                bg="#4a5568", fg="#e2e8f0").pack(anchor="w", padx=8, pady=(3, 0))
        
        self.status_var = tk.StringVar(value="Ready to start detection")
        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                               font=("Segoe UI", 9), bg="#4a5568", fg="#90cdf4", wraplength=300)
        status_label.pack(anchor="w", padx=8, pady=(0, 3))
        
        # # Timer
        # timer_frame = tk.Frame(top_left, bg="#2b6cb8", relief="ridge", bd=1)
        # timer_frame.pack(fill="x", pady=(0, 8))
        
        # tk.Label(timer_frame, text="⏱️ Timer:", font=("Segoe UI", 9, "bold"), 
        #         bg="#2b6cb8", fg="white").pack(anchor="w", padx=8, pady=(3, 0))
        
        # self.timer_var = tk.StringVar(value="Not active")
        # timer_label = tk.Label(timer_frame, textvariable=self.timer_var, 
        #                       font=("Segoe UI", 10, "bold"), bg="#2b6cb8", fg="#ffd700")
        # timer_label.pack(anchor="w", padx=8, pady=(0, 3))
        
        # Replace the old Timer frame logic with this Telemetry section
        telemetry_container = tk.Frame(top_left, bg="#2d3748")
        telemetry_container.pack(fill="x", pady=(0, 8))

        # --- LEFT RECTANGLE: TIMER ---
        timer_frame = tk.Frame(telemetry_container, bg="#2b6cb8", relief="ridge", bd=1)
        timer_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        
        tk.Label(timer_frame, text="⏱️ Timer:", font=("Segoe UI", 8, "bold"), 
                bg="#2b6cb8", fg="white").pack(anchor="w", padx=5, pady=(2, 0))
        
        self.timer_var = tk.StringVar(value="Not active")
        timer_label = tk.Label(timer_frame, textvariable=self.timer_var, 
                              font=("Segoe UI", 9, "bold"), bg="#2b6cb8", fg="#ffd700")
        timer_label.pack(anchor="w", padx=5, pady=(0, 2))

        # --- RIGHT RECTANGLE: LOCATION & DEPTH ---
        env_frame = tk.Frame(telemetry_container, bg="#2d3748", relief="ridge", bd=1)
        env_frame.pack(side="right", fill="both", expand=True, padx=(4, 0))
        
        tk.Label(env_frame, text="📍 Environment:", font=("Segoe UI", 8, "bold"), 
                bg="#2d3748", fg="white").pack(anchor="w", padx=5, pady=(2, 0))
        
        self.env_var = tk.StringVar(value="Waiting for data...")
        env_label = tk.Label(env_frame, textvariable=self.env_var, 
                            font=("Segoe UI", 8), bg="#2d3748", fg="#90cdf4", justify="left")
        env_label.pack(anchor="w", padx=5, pady=(0, 2))

        # Connect buttons
        btn_style = {"font": ("Segoe UI", 9, "bold"), "relief": "flat", "bd": 0, 
                    "pady": 6, "cursor": "hand2"}
        
        btn_connect = tk.Button(top_left, text="📱 Connect Device", 
                               bg="#48bb78", fg="white", activebackground="#38a169",
                               command=self.connect_phone, **btn_style)
        btn_connect.pack(fill="x", pady=(0, 3))
        
        btn_upload = tk.Button(top_left, text="📂 Upload Video", 
                              bg="#4299e1", fg="white", activebackground="#3182ce",
                              command=self.upload_video, **btn_style)
        btn_upload.pack(fill="x", pady=(0, 3))
        
        btn_stop = tk.Button(top_left, text="⏹️ Stop Detection", 
                            bg="#e53e3e", fg="white", activebackground="#c53030",
                            command=self.stop_detection, **btn_style)
        btn_stop.pack(fill="x")
        
        # TOP RIGHT - floating action menu
        top_right = tk.Frame(top_frame, bg="#2d3748", width=180)
        top_right.pack(side="right", fill="both", padx=(15, 0))
        
        # Main action menu button
        self.menu_expanded = False
        self.menu_btn = tk.Button(top_right, text="⚡ Actions ▼", 
                            bg="#4a90e2", fg="white", activebackground="#3182ce",
                            font=("Segoe UI", 10, "bold"), relief="raised", bd=2,
                            pady=8, cursor="hand2", width=15,
                            command=self.toggle_action_menu)
        self.menu_btn.pack(pady=(10, 15))
        
        # Hidden menu container
        self.action_menu = tk.Frame(top_right, bg="#2d3748")
        
        # Compact menu buttons
        menu_style = {"font": ("Segoe UI", 8), "relief": "flat", "bd": 0,
                     "pady": 8, "cursor": "hand2", "width": 15}
        
        tk.Button(self.action_menu, text="📊 Summary", bg="#667eea", fg="white",
                 command=self.show_alert_summary, **menu_style).pack(pady=2)
        tk.Button(self.action_menu, text="💾 Export", bg="#48bb78", fg="white",
                 command=self.export_data, **menu_style).pack(pady=2)
        tk.Button(self.action_menu, text="📧 Email", bg="#4299e1", fg="white",
                 command=self.configure_email, **menu_style).pack(pady=2)
        tk.Button(self.action_menu, text="⚙️ Settings", bg="#a0aec0", fg="white",
                 command=self.open_settings, **menu_style).pack(pady=2)
        
        # BOTTOM HALF - graph and results
        bottom_frame = tk.Frame(content_frame, bg="#2d3748")
        bottom_frame.pack(fill="both", expand=True)
        
        # Detection graph
        graph_label = tk.Label(bottom_frame, text="📊 Detection Graph", 
                              font=("Segoe UI", 11, "bold"), bg="#2d3748", fg="#e2e8f0")
        graph_label.pack(anchor="w", pady=(0, 5))
        
        graph_container = tk.Frame(bottom_frame, bg="#1a202c", relief="sunken", bd=2, height=200)
        graph_container.pack(fill="x", pady=(0, 5))
        graph_container.pack_propagate(False)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 2), dpi=80, facecolor='#1a202c')
        self.ax = self.fig.add_subplot(111, facecolor='#1a202c')
        self.ax.set_xlabel('Time', color='white', fontsize=8)
        self.ax.set_ylabel('Objects', color='white', fontsize=8)
        self.ax.tick_params(colors='white', labelsize=7)
        self.ax.grid(True, alpha=0.3, color='white')
        
        self.graph_canvas = FigureCanvasTkAgg(self.fig, graph_container)
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Detection results
        results_label = tk.Label(bottom_frame, text="🔍 Detection Results", 
                                font=("Segoe UI", 11, "bold"), bg="#2d3748", fg="#e2e8f0")
        results_label.pack(anchor="w", pady=(0, 5))
        
        results_frame = tk.Frame(bottom_frame, bg="#1a202c", relief="sunken", bd=2, height=100)
        results_frame.pack(fill="both", expand=True)
        results_frame.pack_propagate(False)
        
        self.results_text = tk.Text(results_frame, bg="#1a202c", fg="#e2e8f0", 
                                   font=("Consolas", 8), relief="flat", bd=3,
                                   selectbackground="#4a90e2", insertbackground="white",
                                   wrap="word")
        
        scrollbar = tk.Scrollbar(results_frame, bg="#2d3748")
        scrollbar.pack(side="right", fill="y")
        self.results_text.pack(side="left", fill="both", expand=True)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
        # Setup close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def toggle_action_menu(self):
        """Toggle the action menu visibility"""
        if self.menu_expanded:
            self.action_menu.pack_forget()
            self.menu_expanded = False
            self.menu_btn.config(text="⚡ Actions ▼")
        else:
            self.action_menu.pack(after=self.menu_btn, fill="x", pady=(0, 10))
            self.menu_expanded = True
            self.menu_btn.config(text="⚡ Actions ▲")
    
    def on_detection_change(self, event=None):
        """Handle detection type change"""
        detection_type = self.detection_var.get()
        self.backend.reset_video()
        model_changed = self.backend.load_model(detection_type)
        if model_changed:
            self.status_var.set(f"Switched to {detection_type} - Ready to start")
            self.results_text.delete("1.0", tk.END)
            self.canvas.delete("all")
    
    def connect_phone(self):
        """Handle phone connection button"""
        # Check if detection is already running
        if self.backend.is_detecting():
            result = messagebox.askyesno(
                "Detection Running", 
                "Detection is currently running. Do you want to stop the current detection and connect to device?",
                icon="warning"
            )
            if not result:
                return
            self.backend.reset_video()
            self.canvas.delete("all")
            self.results_text.delete("1.0", tk.END)
        
        detection_type = self.detection_var.get()
        
        # Ensure model is loaded
        self.status_var.set("Loading model...")
        self.root.update()
        
        if not self.backend.load_model(detection_type):
            messagebox.showerror("Error", f"Failed to load {detection_type} model")
            self.status_var.set("Model loading failed")
            return
        
        # Connect to phone with progress updates
        self.status_var.set("Connecting to device...")
        self.root.update()
        
        # Use after_idle to prevent UI blocking
        self.root.after_idle(self._do_phone_connection)
    
    def _do_phone_connection(self):
        """Perform phone connection in background"""
        try:
            success, message = self.backend.connect_phone()
            self.status_var.set(message)
            
            if success:
                self.detection_start_time = datetime.now()
                self.start_detection_timer()
                self.root.after(100, self.update_frame)
            else:
                # Show error dialog for connection failures
                if "timeout" in message.lower() or "failed" in message.lower():
                    messagebox.showwarning("Connection Failed", message)
        except Exception as e:
            self.status_var.set(f"Connection error: {str(e)}")
            messagebox.showerror("Error", f"Unexpected error during connection: {str(e)}")
    
    def stop_detection(self):
        """Stop current detection"""
        if not self.backend.is_detecting():
            messagebox.showinfo("No Detection", "No detection is currently running.")
            return
        
        self.backend.reset_video()
        self.canvas.delete("all")
        self.results_text.delete("1.0", tk.END)
        self.cancel_detection_timer()
        self.status_var.set("Detection stopped by user")
        messagebox.showinfo("Detection Stopped", "Detection has been stopped successfully.")
    
    def upload_video(self):
        """Handle video upload button"""
        # Check if detection is already running
        if self.backend.is_detecting():
            result = messagebox.askyesno(
                "Detection Running", 
                "Detection is currently running. Do you want to stop the current detection and upload a new video?",
                icon="warning"
            )
            if not result:
                return
            self.backend.reset_video()
            self.canvas.delete("all")
            self.results_text.delete("1.0", tk.END)
        
        detection_type = self.detection_var.get()
        
        # Ensure model is loaded
        self.status_var.set("Loading model...")
        self.root.update()
        
        if not self.backend.load_model(detection_type):
            messagebox.showerror("Error", f"Failed to load {detection_type} model")
            self.status_var.set("Model loading failed")
            return
        
        # Automatic video selection based on detection type (COMMENTED)
        # if detection_type == "Asset Detection":
        #     file_path = r"C:\Users\HP\Documents\FYP DATASET\camera_stick.MOV"
        # elif detection_type == "Fish Detection":
        #     file_path = r"C:\Users\HP\Documents\FYP DATASET\fish (2).MOV"
        # else:
        #     file_path = None
        
        # Manual file selection (CURRENT)
        file_path = filedialog.askopenfilename(
            title=f"Select video for {detection_type}",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not file_path:
            self.status_var.set("Ready to start detection")
            return
            
        self.status_var.set("Loading video...")
        self.root.update()
        
        success, message = self.backend.load_video(file_path)
        if not success:
            messagebox.showerror("Error", message)
            self.status_var.set("Video loading failed")
            return
            
        self.status_var.set(f"{detection_type}: Starting video playback")
        self.root.update()
        self.detection_start_time = datetime.now()
        self.start_detection_timer()
        self.root.after(100, self.update_frame)  # Small delay before starting
    
    def update_frame(self):
        """Update the video frame display"""
        if not self.backend.running_source:
            return
            
        frame = self.backend.get_frame()
        
        # Get location info if connected to phone
        loc_text = "N/A"
        if self.backend.running_source == "phone":
            try:
                _, loc_raw = self.backend.get_phone_location()
                # Clean up the message for the small box
                loc_text = loc_raw.replace("Lat: ", "L:").replace("Lon: ", " Lo:")
            except Exception as e:
                loc_text = "N/A"
        
        if not frame:
            if self.backend.running_source == "video":
                self.status_var.set("Video playback completed")
                self.backend.reset_video()
                return
            self.root.after(100, self.update_frame)
            return
        
        # Run detection (YOLO or Fish)
        detection_result = self.backend.run_inference(frame)
        if len(detection_result) == 5:  # Fish detection returns 5 values
            frame, detected, anomalies, fish_count, depth = detection_result
        else:  # Asset detection returns 4 values
            frame, detected, anomalies, depth = detection_result
            fish_count = None
        
        # Update environment telemetry
        depth_str = f"{depth}m" if depth != "Unknown" else "---"
        self.env_var.set(f"Depth: {depth_str}\n{loc_text}")

        active_anomalies = self.backend.anomaly_detector.get_active_anomalies()
        
        # Update results panel with better formatting
        if detected or active_anomalies:
            self.results_text.delete("1.0", tk.END)
            # self.results_text.insert(tk.END, f"\n🌊 Depth: {depth} meters\n\n")
            detection_type = self.detection_var.get()
            self.results_text.insert(tk.END, f"=== {detection_type} Results ===\n\n")
            
            
            
            # Show active anomalies first (persistent display)
            if active_anomalies:
                context = "Asset" if detection_type == "Asset Detection" else "Fish"
                self.results_text.insert(tk.END, f"🚨 {context.upper()} MONITORING ANOMALIES:\n")
                for anomaly in active_anomalies:
                    severity_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
                    icon = severity_color.get(anomaly['severity'], "⚪")
                    self.results_text.insert(tk.END, f"{icon} [{anomaly['timestamp']}] {anomaly['severity']}: {anomaly['detection']}\n")
                self.results_text.insert(tk.END, "\n")
                
                # Trigger alerts for new anomalies only
                for anomaly in anomalies:
                    self.backend.anomaly_detector.trigger_alert(anomaly)
            
            # Show all detections
            if detected:
                self.results_text.insert(tk.END, "📊 All Detections:\n")
                for i, detection in enumerate(detected, 1):
                    self.results_text.insert(tk.END, f"{i:2d}. {detection}\n")
            
            status_msg = f"Detecting... Found {len(detected)} objects"
            if active_anomalies:
                status_msg += f" | {len(active_anomalies)} ACTIVE ANOMALIES!"
            self.status_var.set(status_msg)
        else:
            self.status_var.set("Detecting... No objects found")
        
        # Display frame on canvas
        self.display_frame(frame)
        
        # Update graph with appropriate data
        if self.detection_var.get() == "Fish Detection" and fish_count is not None:
            self.update_detection_graph(fish_count, is_fish_count=True)
        else:
            self.update_detection_graph(len(detected), is_fish_count=False)
        
        self.root.after(33, self.update_frame)  # ~30 FPS
    
    def display_frame(self, frame):
        """Display frame on canvas with proper scaling and border"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Crop frame to remove phone status bars (top 10% and bottom 10%)
        if self.backend.running_source == "video":
            crop_top = int(frame.height * 0.05)  # Remove top 10%
            crop_bottom = int(frame.height * 0.95)  # Remove bottom 10%
            frame = frame.crop((0, crop_top, frame.width, crop_bottom))
        
        # Scale image to fit canvas (preserve aspect ratio)
        frame_aspect = frame.width / frame.height
        canvas_aspect = canvas_width / canvas_height
        
        if frame_aspect > canvas_aspect:
            new_width = canvas_width - 20  # padding
            new_height = int(new_width / frame_aspect)
        else:
            new_height = canvas_height - 20  # padding
            new_width = int(new_height * frame_aspect)
        
        frame_resized = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(frame_resized)
        
        self.canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Add subtle border around image
        self.canvas.create_rectangle(x-2, y-2, x+new_width+2, y+new_height+2, 
                                   outline="#4a90e2", width=2)
        
        self.image_item = self.canvas.create_image(x, y, anchor="nw", image=img_tk)
        self.canvas.image = img_tk
    
    def on_close(self):
        """Handle window close event"""
        self.cancel_detection_timer()
        self.backend.cleanup()
        self.root.destroy()
    
    def show_alert_summary(self):
        """Show alert summary dialog"""
        summary = self.backend.anomaly_detector.get_alert_summary()
        
        summary_text = f"""Recent Activity Summary (Last 5 minutes):
        
📊 Total Detections: {summary['total_detections']}
🏢 Asset Detections: {summary['asset_detections']}
🐟 Fish Detections: {summary['fish_detections']}

🕰️ Last Detection: {datetime.fromtimestamp(summary['last_detection']).strftime('%H:%M:%S') if summary['last_detection'] else 'None'}
        
✅ Asset Monitoring Expected:
Normal: empty water tank, structures, rope, lying oil rig, lying pipe, protrude pipe
Anomalies: major shift (CRITICAL), fish/fishing pond (HIGH), moving camera stick (MEDIUM)

🐟 Fish Monitoring Expected:
Normal: fish, fishing pond
Anomalies: lying oil rig/pipe (CRITICAL), major shift/protrude pipe (HIGH), moving camera stick/structures (MEDIUM), rope (LOW)"""
        
        messagebox.showinfo("Alert Summary", summary_text)
    
    def export_data(self):
        """Export detection data"""
        try:
            filepath = self.backend.anomaly_detector.export_session_data()
            messagebox.showinfo("Export Complete", f"Session data exported to:\n{filepath}\n\nDaily CSV logs are saved in the 'logs' folder.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
    
    def configure_email(self):
        """Configure email alerts"""
        config_window = tk.Toplevel(self.root)
        config_window.title("Email Alert Configuration")
        config_window.geometry("400x500")
        config_window.configure(bg="#2d3748")
        
        # Email configuration form
        tk.Label(config_window, text="Email Alert Configuration", font=("Segoe UI", 14, "bold"), 
                bg="#2d3748", fg="white").pack(pady=10)
        
        # SMTP Server
        tk.Label(config_window, text="SMTP Server:", bg="#2d3748", fg="white").pack(anchor="w", padx=20)
        smtp_entry = tk.Entry(config_window, width=40)
        smtp_entry.insert(0, "smtp.gmail.com")
        smtp_entry.pack(padx=20, pady=5)
        
        # SMTP Port
        tk.Label(config_window, text="SMTP Port:", bg="#2d3748", fg="white").pack(anchor="w", padx=20)
        port_entry = tk.Entry(config_window, width=40)
        port_entry.insert(0, "587")
        port_entry.pack(padx=20, pady=5)
        
        # Sender Email
        tk.Label(config_window, text="Sender Email:", bg="#2d3748", fg="white").pack(anchor="w", padx=20)
        sender_entry = tk.Entry(config_window, width=40)
        sender_entry.insert(0,"jocelynngieng@gmail.com")
        sender_entry.pack(padx=20, pady=5)
        
        # Sender Password
        tk.Label(config_window, text="App Password:", bg="#2d3748", fg="white").pack(anchor="w", padx=20)
        password_entry = tk.Entry(config_window, width=40, show="*")
        password_entry.insert(0,"zpmlijmfrtgtpmei")
        password_entry.pack(padx=20, pady=5)
        
        # Recipients
        tk.Label(config_window, text="Recipients (comma-separated):", bg="#2d3748", fg="white").pack(anchor="w", padx=20)
        recipients_entry = tk.Entry(config_window, width=40)
        recipients_entry.insert(0,"jocelynngieng@gmail.com")
        recipients_entry.pack(padx=20, pady=5)
        
        # Info label
        info_text = "Note: For Gmail, use App Password instead of regular password.\nOnly CRITICAL and HIGH alerts will be emailed."
        tk.Label(config_window, text=info_text, bg="#2d3748", fg="#90cdf4", 
                wraplength=350, justify="left").pack(pady=10)
        
        def save_config():
            try:
                smtp_server = smtp_entry.get().strip()
                smtp_port = int(port_entry.get().strip())
                sender_email = sender_entry.get().strip()
                sender_password = password_entry.get().strip()
                recipients = [email.strip() for email in recipients_entry.get().split(',') if email.strip()]
                
                if not all([smtp_server, smtp_port, sender_email, sender_password, recipients]):
                    messagebox.showerror("Error", "Please fill in all fields")
                    return
                
                self.backend.anomaly_detector.configure_email_alerts(
                    smtp_server, smtp_port, sender_email, sender_password, recipients
                )
                
                messagebox.showinfo("Success", "Email configuration saved successfully!")
                config_window.destroy()
                
            except ValueError:
                messagebox.showerror("Error", "Invalid port number")
            except Exception as e:
                messagebox.showerror("Error", f"Configuration error: {str(e)}")
        
        def test_email():
            save_config()
            success, message = self.backend.anomaly_detector.test_email_configuration()
            if success:
                messagebox.showinfo("Test Email", "Test email sent successfully!")
            else:
                messagebox.showerror("Test Email Failed", message)
        
        # Buttons
        btn_frame = tk.Frame(config_window, bg="#2d3748")
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="Save Config", command=save_config, 
                 bg="#48bb78", fg="white", padx=20).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Test Email", command=test_email, 
                 bg="#4299e1", fg="white", padx=20).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=config_window.destroy, 
                 bg="#e53e3e", fg="white", padx=20).pack(side="left", padx=5)
    
    def start_detection_timer(self):
        """Start detection duration timer"""
        if self.detection_duration > 0:
            self.timer_job = self.root.after(self.detection_duration * 1000, self.stop_detection_timer)
            self.update_timer_display()
        else:
            self.timer_var.set("∞ Unlimited")
    
    def stop_detection_timer(self):
        """Stop detection and clear feed when timer expires"""
        self.backend.reset_video()
        self.canvas.delete("all")
        self.results_text.delete("1.0", tk.END)
        self.status_var.set("Detection completed - Timer expired")
        self.timer_var.set("Completed")
        messagebox.showinfo("Detection Complete", f"Detection stopped after {self.detection_duration} seconds")
    
    def cancel_detection_timer(self):
        """Cancel active detection timer"""
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        self.timer_var.set("Not active")
    
    def update_timer_display(self):
        """Update timer countdown display"""
        if self.detection_start_time and self.backend.is_detecting() and self.detection_duration > 0:
            elapsed = (datetime.now() - self.detection_start_time).total_seconds()
            remaining = max(0, self.detection_duration - elapsed)
            
            if remaining > 0:
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                self.timer_var.set(f"{minutes:02d}:{seconds:02d} remaining")
                # Update every second
                self.root.after(1000, self.update_timer_display)
            else:
                self.timer_var.set("00:00 remaining")
        elif self.backend.is_detecting() and self.detection_duration == 0:
            elapsed = (datetime.now() - self.detection_start_time).total_seconds()
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_var.set(f"∞ Running {minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer_display)
        else:
            self.timer_var.set("Not active")
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Detection Settings")
        settings_window.geometry("450x500")
        settings_window.configure(bg="#2d3748")
        settings_window.resizable(True, True)
        
        # Settings title
        tk.Label(settings_window, text="Detection Settings", font=("Segoe UI", 14, "bold"), 
                bg="#2d3748", fg="white").pack(pady=15)
        
        # Detection duration setting
        duration_frame = tk.Frame(settings_window, bg="#2d3748")
        duration_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(duration_frame, text="Detection Duration:", font=("Segoe UI", 11, "bold"), 
                bg="#2d3748", fg="white").pack(anchor="w")
        
        duration_var = tk.StringVar(value=str(self.detection_duration) if self.detection_duration > 0 else "0")
        duration_entry = tk.Entry(duration_frame, textvariable=duration_var, font=("Segoe UI", 11), width=10)
        duration_entry.pack(anchor="w", pady=5)
        
        tk.Label(duration_frame, text="(0 = Unlimited, >0 = Seconds to auto-stop)", 
                font=("Segoe UI", 9), bg="#2d3748", fg="#90cdf4").pack(anchor="w")
        
        # Preset buttons
        preset_frame = tk.Frame(settings_window, bg="#2d3748")
        preset_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(preset_frame, text="Quick Presets:", font=("Segoe UI", 11, "bold"), 
                bg="#2d3748", fg="white").pack(anchor="w")
        
        preset_btn_frame = tk.Frame(preset_frame, bg="#2d3748")
        preset_btn_frame.pack(fill="x", pady=5)
        
        def set_preset(seconds):
            duration_var.set(str(seconds))
        
        tk.Button(preset_btn_frame, text="30s", command=lambda: set_preset(30), 
                 bg="#4299e1", fg="white", width=6).pack(side="left", padx=2)
        tk.Button(preset_btn_frame, text="1min", command=lambda: set_preset(60), 
                 bg="#4299e1", fg="white", width=6).pack(side="left", padx=2)
        tk.Button(preset_btn_frame, text="5min", command=lambda: set_preset(300), 
                 bg="#4299e1", fg="white", width=6).pack(side="left", padx=2)
        tk.Button(preset_btn_frame, text="∞", command=lambda: set_preset(0), 
                 bg="#48bb78", fg="white", width=6).pack(side="left", padx=2)
        
        # Current status
        status_frame = tk.Frame(settings_window, bg="#4a5568", relief="ridge", bd=1)
        status_frame.pack(pady=15, padx=20, fill="x")
        
        tk.Label(status_frame, text="Current Status:", font=("Segoe UI", 10, "bold"), 
                bg="#4a5568", fg="white").pack(anchor="w", padx=10, pady=(5, 0))
        
        if self.detection_start_time and self.backend.is_detecting():
            elapsed = (datetime.now() - self.detection_start_time).seconds
            remaining = self.detection_duration - elapsed if self.detection_duration > 0 else "∞"
            status_text = f"Running: {elapsed}s elapsed, {remaining}s remaining"
        else:
            status_text = "Not detecting"
        
        tk.Label(status_frame, text=status_text, font=("Segoe UI", 9), 
                bg="#4a5568", fg="#90cdf4").pack(anchor="w", padx=10, pady=(0, 5))
        
        # Buttons
        btn_frame = tk.Frame(settings_window, bg="#2d3748")
        btn_frame.pack(pady=20)
        
        def save_settings():
            try:
                new_duration = int(duration_var.get())
                if new_duration < 0:
                    messagebox.showerror("Error", "Duration must be 0 or positive")
                    return
                
                self.detection_duration = new_duration
                messagebox.showinfo("Success", f"Detection duration set to {new_duration} seconds\n(0 = unlimited)")
                settings_window.destroy()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")
        
        tk.Button(btn_frame, text="Save", command=save_settings, 
                 bg="#48bb78", fg="white", padx=20).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=settings_window.destroy, 
                 bg="#e53e3e", fg="white", padx=20).pack(side="left", padx=5)
    
    def update_detection_graph(self, count, is_fish_count=False):
        """Update real-time detection graph"""
        current_time = datetime.now()
        self.detection_times.append(current_time)
        
        if is_fish_count:
            self.fish_counts.append(count)
            # Keep detection_counts in sync
            self.detection_counts.append(0)
        else:
            self.detection_counts.append(count)
            # Keep fish_counts in sync
            self.fish_counts.append(0)
        
        # Update graph every 10 frames to avoid lag
        if len(self.detection_times) % 10 == 0:
            self.ax.clear()
            self.ax.set_facecolor('#1a202c')
            
            if is_fish_count:
                # Show fish count graph
                self.ax.plot(list(self.detection_times), list(self.fish_counts), 
                            color='#00ffff', linewidth=2, marker='o', markersize=3, label='Fish Count')
                self.ax.set_ylabel('Fish Count', color='white')
                max_val = max(self.fish_counts) if self.fish_counts else 10
            else:
                # Show object detection graph
                self.ax.plot(list(self.detection_times), list(self.detection_counts), 
                            color='#00ff00', linewidth=2, marker='o', markersize=3, label='Objects')
                self.ax.set_ylabel('Objects Detected', color='white')
                max_val = max(self.detection_counts) if self.detection_counts else 10
            
            self.ax.set_xlabel('Time', color='white')
            self.ax.tick_params(colors='white')
            self.ax.grid(True, alpha=0.3, color='white')
            self.ax.set_ylim(0, max(10, max_val + 2))
            
            # Format x-axis to show time
            if len(self.detection_times) > 1:
                self.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                self.fig.autofmt_xdate()
            
            self.graph_canvas.draw()
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()