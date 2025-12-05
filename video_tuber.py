import os
import time
import random
#module for detection of microphone input
import sounddevice as sd
import numpy as np
#module for creating the window and reading video files
import cv2
#module needed for creating queues to store incomig data from sockets
import queue
import socket
import threading

# ---------------- CONFIG ----------------
### Screen
WINDOW_NAME = "Camera :3"
SCREEN_WIDTH = 350
SCREEN_HEIGHT = 350
### RULES
###### MIC 
MIC_DEVICE_INDEX = 1
######### Options when detecting noise
AUDIO_THRESHOLD_NOISE = 0.2
NOISE_DURATION = 0.0
######### Options when detecting silence
AUDIO_THRESHOLD_SILENCE = 0.2
SILENCE_DURATION = 1.0
###### MIDI
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000       # Port to listen on



### FRAMES
VIDEO_END_CUTOFF = 20  # Number of frames before the actual end to consider the video finished


### FILTERS
###### TRANSITION FILTERS
TRANSITION_FILTER_FRAMES = 5    # Number of frames glitch runs during a transition
######### GLITCH
GLITCH_ENABLE = True             # Enable/disable glitch filter
GLITCH_SHIFT = 25               # Max horizontal shift for glitch bars
GLITCH_BAR_MIN = 5              # Minimum number of glitch bars per frame
GLITCH_BAR_MAX = 10             # Maximum number of glitch bars per frame
BLUE_BOOST = 80                 # Intensity boost for blue channel in glitch
###### REGULAR FILTERS
######### VHS wobble
ENABLE_VHS = True
VHS_AMPLITUDE = 2
VHS_FREQ = 10.0
######### Scanlines
SCANLINE_ENABLE = True
SCANLINE_OPACITY = 160
SCANLINE_SPACING = 4
######### Chromatic aberration
ENABLE_CA = True
CA_SHIFT = 4

### Global Variables
FRAME_ENDED = False
video_requests = queue.Queue()
# ---------------- STATE STRUCTURE ----------------
class StateStruct:
    def __init__(self, name, video_random, videos=None, transitions=None):
        self.name = name
        self.videos = videos if videos else []
        self.video_random = video_random
        # transitions: list of tuples (next_state_name, rule_name, config_tuple)
        self.transitions = transitions if transitions else []

    def __repr__(self):
        return (
            f"StateStruct(name={self.name!r}, "
            f"videos={self.videos!r}, "
            f"video_random={self.video_random!r}, "
            f"transitions={self.transitions!r})"
        )

# ---------------- DEFINE STATES ----------------
STATES = {
    "Idle": StateStruct(
        name="Idle",
        video_random=True,
        transitions=[("Talking", "MIC", (AUDIO_THRESHOLD_NOISE, NOISE_DURATION, "POSITIVE")),
                     ("Emotes", "MIDI", (None))]
    ),
    "Talking": StateStruct(
        name="Talking",
        video_random=True,
        transitions=[("Idle", "MIC", (AUDIO_THRESHOLD_SILENCE, SILENCE_DURATION, "NEGATIVE"))]
    ),
    "Emotes": StateStruct(
        name="Emotes",
        video_random=False,
        transitions=[("Idle", "Inactivity", (None))]
    ),
}

# ---------------- AUTO-LOAD VIDEOS ----------------

def auto_load_videos_into_states(state_map):
    """
    Scans the subfolder with the same name as the state for video files.
    Example: folder 'Idle' contains 'Idle_1.mp4', 'Idle_hi.mov', etc.
    """
    video_ext = (".mp4", ".mov", ".avi", ".mkv")

    for state_name, state_struct in state_map.items():
        folder_path = os.path.join(os.getcwd(), state_name)
        matched_files = []

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(video_ext):
                    matched_files.append(os.path.join(folder_path, file))
        else:
            print(f"Warning: folder '{folder_path}' does not exist.")
            exit

        state_struct.videos = matched_files


# ---------------- Filters ---------------------
class Filters:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Transition glitch variables
        self.transition_filter_active = False
        self.transition_filter_frames_remaining = 0
        self.TRANSITION_FILTER_TOTAL_FRAMES = TRANSITION_FILTER_FRAMES

        # Last clean frame for glitch effect
        self.LAST_CLEAN_FRAME = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # -------- Glitch --------
    def generate_glitch_frame(self, base_frame):
        base = base_frame.copy()
        self.LAST_CLEAN_FRAME = base_frame.copy()
        num_bars = random.randint(GLITCH_BAR_MIN, GLITCH_BAR_MAX)

        for _ in range(num_bars):
            y = random.randint(0, self.screen_height - 2)
            h = random.randint(1, min(10, self.screen_height - y))
            shift = random.randint(-GLITCH_SHIFT, GLITCH_SHIFT)

            # Red channel
            x_r = max(0, shift)
            base[y:y+h, x_r:self.screen_width, 0] = 255

            # Green channel
            x_g = max(0, -shift)
            base[y:y+h, x_g:self.screen_width, 1] = 255

            # Blue channel boost
            blue = base[y:y+h, :, 2].astype(np.int16) + BLUE_BOOST
            base[y:y+h, :, 2] = np.clip(blue, 0, 255).astype(np.uint8)

        return base

    # -------- Scanlines --------
    def apply_scanlines(self, frame):
        if not SCANLINE_ENABLE:
            return frame
        out = frame.copy()
        for y in range(0, out.shape[0], SCANLINE_SPACING):
            darkened = out[y:y+1].astype(np.int16) - SCANLINE_OPACITY
            out[y:y+1] = np.clip(darkened, 0, 255).astype(np.uint8)
        return out

    # -------- Chromatic Aberration --------
    def apply_chromatic_aberration(self, frame):
        if not ENABLE_CA:
            return frame
        b, g, r = cv2.split(frame)
        r_shift = np.roll(r, CA_SHIFT, axis=1)
        g_shift = np.roll(g, -CA_SHIFT, axis=0)
        return cv2.merge([b, g_shift, r_shift])

    # -------- VHS Wobble --------
    def apply_vhs_wobble(self, frame):
        if not ENABLE_VHS:
            return frame
        h = frame.shape[0]
        t = time.time()
        out = np.empty_like(frame)
        rows = np.arange(h)
        shifts = (VHS_AMPLITUDE * np.sin(rows / VHS_FREQ + t * 8.0)).astype(np.int32)
        for i, s in enumerate(shifts):
            out[i] = np.roll(frame[i], s, axis=0) if s != 0 else frame[i]
        return out

    # -------- Apply All Filters --------
    def apply_filters(self, frame):
        if GLITCH_ENABLE and self.transition_filter_active:
            frame = self.generate_glitch_frame(frame)
            self.transition_filter_frames_remaining -= 1
            if self.transition_filter_frames_remaining <= 0:
                self.transition_filter_active = False

        # Apply other visual filters
        frame = self.apply_vhs_wobble(frame)
        frame = self.apply_chromatic_aberration(frame)
        frame = self.apply_scanlines(frame)

        return frame


    def start_transition_filter(self):
        self.transition_filter_active = True
        self.transition_filter_frames_remaining = self.TRANSITION_FILTER_TOTAL_FRAMES



# ---------------- VIDEO PLAYER ----------------
class VideoPlayer:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.current_video = None
        self.cap = None

    def select_new_video(self, new_video_requested):
        
        self.current_video = new_video_requested
        print(f"new_video_requested 2: {self.current_video}")
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.current_video)
        print(f"Selected video: {self.current_video}")

    def select_random_video(self, video_list):
        if video_list:
            self.current_video = random.choice(video_list)
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.current_video)
            print(f"Selected video: {self.current_video}")
        else:
            self.current_video = None
            self.cap = None
            print("No videos available to play.")

    def get_frame(self):
        global FRAME_ENDED
        if not self.cap:
            return None

        frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #Peek at the next frame without advancing: use cap.get to detect if near end
        near_end = total_frames > 0 and frame_idx >= total_frames - VIDEO_END_CUTOFF

        ret, frame = self.cap.read()

        if not ret or near_end:
            #Trigger transition filter
            self.start_transition_filter()
            #Select another random video
            self.select_random_video(self.current_state.videos)
            #Read the first frame of the new video
            ret, frame = self.cap.read()
            FRAME_ENDED = False

        if near_end:
            FRAME_ENDED = True

        if frame is not None:
            frame = cv2.resize(frame, (self.screen_width, self.screen_height))
        return frame



    def release(self):
        if self.cap:
            self.cap.release()

# ---------------- STATE MACHINE ----------------
class StateMachine(VideoPlayer, Filters):
    def __init__(self, states, initial_state_name="Idle",
                 screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT):
        VideoPlayer.__init__(self, screen_width, screen_height)
        Filters.__init__(self, screen_width, screen_height)
        self.states = states
        self.video_random = False
        self.current_state = states[initial_state_name]
        self.new_video_requested = "none"

        # Pick initial video
        self.select_random_video(self.current_state.videos)

    def update(self):
        #Iterate through the transitions list in the current state and get the rules of transitioning
        for next_state_name, rule_name, config in self.current_state.transitions:
            #Iterate through the rules list and get the rules names and callback
            for r_name, init_fn, callback_fn in RULES:
                #If a rule for transition matches the one in the rules list
                if r_name == rule_name:
                    #Call the rule callback to see if the transition rule applies
                    if config is None:
                       result = callback_fn()
                    else: 
                        result = callback_fn(*config)
                    #Check if the transition rule applies
                    if result:
                        #Trigger transition filter
                        self.start_transition_filter()
                        #Switch to the next state
                        self.switch_state(next_state_name)

    def switch_state(self, new_state_name):
        #Validate that the new state is valid
        if new_state_name in self.states:
            #Set the current state as the new state
            self.current_state = self.states[new_state_name]
            print(f"Switched to state: {new_state_name}")
            print(f"video_random: {self.current_state.video_random}")
            if self.current_state.video_random is True:
                #Load new video from new current state
                self.select_random_video(self.current_state.videos)
            else:
                #Load specific video requested
                print(f"new_video_requested: {self.new_video_requested}")
                self.select_new_video(self.new_video_requested)
        #The new state was not found. Go back to idle state as default
        else:
            print(f"State {new_state_name} not found. Switching to Idle state")
            #Set idle state as default
            self.current_state = "Idle"
            #Select video from idle state
            self.select_random_video(self.current_state.videos)

    def request_new_video(self, new_video):
        self.new_video_requested = new_video



# ---------------- RULES ----------------
### MIC
SOUND_DETECTED = False
LAST_NOISE_TIME = 0.0
VOLUME = 0.0

###### Input Stream callback
#This function is needed to update the detected volume level by Input Stream
def InputStream_callback(indata, frames, time_info, status):
    global VOLUME
    #Save the volume level to a shared variable so the rule callback can access this value
    VOLUME = np.linalg.norm(indata)

###### INIT 
def mic_init():
    #Start the Input Stream volume detection
    sd.InputStream(device=MIC_DEVICE_INDEX, channels=1, callback=InputStream_callback).start()

###### CALLBACK
def mic_callback(threshold, duration, threshold_type):
    global SOUND_DETECTED, LAST_NOISE_TIME, VOLUME
    result = False
    threshold_passed = False

    #Check the type of threshold that we need to use
    if (threshold_type == "POSITIVE"):
        #Check if the volume has passed the positive threshold
        if (VOLUME >= threshold):
            threshold_passed = True
    elif (threshold_type == "NEGATIVE"):
        #Check if the volume has passed the negative threshold
        if (VOLUME <= threshold):
            threshold_passed = True
    

    #If the threshold has passed
    if (threshold_passed):
        #Get current time
        time_now = time.time()
        #If this is the fist time detecting time
        if (SOUND_DETECTED is False):
            #Set sound detected flag
            SOUND_DETECTED = True
            #Record time when sound was detected
            LAST_NOISE_TIME = time.time()
        #Sound was already detected
        else: 
            #Check if sound has been going longer than rule's duration
            if ((time_now - LAST_NOISE_TIME) > duration):
                #Rule is valid
                result = True
                #Reset sound detected flag
                SOUND_DETECTED = False
    else:
        #Clear flag as sound is no longer detected
        SOUND_DETECTED = False
    
    return result

### Inactivity
###### INIT 
def inactivity_init():
    pass

###### CALLBACK
def inactivity_callback():
    global FRAME_ENDED
    result = FRAME_ENDED  
    return result

### MIDI
###### Socket callback
def handle_client(client_socket, address):
    print(f"New connection from {address}")
    with client_socket:
        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break

                message = data.decode('utf-8').strip()
                print(f"[{address}] Received raw: {message}")

                # Split by comma
                parts = message.split(",", 1)   # split only once
                first_param = parts[0].strip()

                print(f"[{address}] Parsed video request: {first_param}")

                # PUSH only first param to queue
                video_requests.put(first_param)

            except ConnectionResetError:
                break

    print(f"Connection closed: {address}")

###### MIDI Server Thread
def midi_server_thread(server):
    while True:
        client_socket, address = server.accept()
        thread = threading.Thread(target=handle_client, args=(client_socket, address), daemon=True)
        thread.start()

###### INIT 
def midi_init():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    print(f"MIDI Server listening on {HOST}:{PORT}")

    # Run server loop in its own thread
    thread = threading.Thread(target=midi_server_thread, args=(server,), daemon=True)
    thread.start()

###### CALLBACK
def midi_callback():
    result = False
    try:
        # Non-blocking pop
        new_video = video_requests.get_nowait()
        result = True
    except queue.Empty:
        return result  # Nothing to process

    # Do whatever you want with it
    sm.request_new_video = new_video
    return result


### List of rules
RULES = [
    ("MIC", mic_init, mic_callback),
    ("Inactivity", inactivity_init, inactivity_callback),
    ("MIDI", midi_init, midi_callback),
]

# ---------------- TEST ----------------
if __name__ == "__main__":
    # Load the videos
    auto_load_videos_into_states(STATES)

    sm = StateMachine(STATES)

    # Initialize all rules
    for rule_name, init_fn, callback_fn in RULES:
        # Run the init function for the particular rule
        init_fn() 

    # Create window to display the frames
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # Resize the window
    cv2.resizeWindow(WINDOW_NAME, SCREEN_WIDTH, SCREEN_HEIGHT)

    # Main loop
    while True:
        # Update the state machine
        sm.update()
        # Get new frame
        frame = sm.get_frame()
        # Apply any filters to frame
        frame = sm.apply_filters(frame)

        # Display frame only if it is valid
        if frame is not None:
            cv2.imshow(WINDOW_NAME, frame)

        # Check if user has pressed the esc key to close the program
        if cv2.waitKey(30) & 0xFF == 27:
            break

