from pathlib import Path
import threading

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)



# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "xgboost_digit_model.pkl"


# ============================================================
# 3. MODEL CONSTANTS
# ============================================================

MODEL_WIDTH = 28
MODEL_HEIGHT = 28
EXPECTED_FEATURES = 784


# ============================================================
# 4. AIR DRAWING CONSTANTS
# ============================================================

SMOOTHING_ALPHA = 0.35

DRAW_STABILITY_FRAMES = 2

GESTURE_LOSS_TOLERANCE = 3

CLEAR_HOLD_FRAMES = 18

DRAW_THICKNESS = 12


# ============================================================
# 5. CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #244d73;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1050px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #163f63;
        border-right: 1px solid #23e8e8;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .sidebar-brand {
        font-size: 18px;
        font-weight: 900;
        margin-top: 10px;
        margin-bottom: 22px;
    }

    .online-box {
        background: #155b73;
        padding: 16px;
        border-radius: 10px;
        font-weight: 800;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 16px;
        font-weight: 900;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    .sidebar-text {
        font-size: 13px;
        line-height: 1.7;
    }

    .small-muted {
        color: #b5cadb !important;
        font-size: 12px;
    }

    .best-model-box {
        background: #155b73;
        padding: 14px;
        border-radius: 10px;
        margin-top: 14px;
        font-weight: 800;
    }

    .hero-title {
        color: white;
        font-size: 38px;
        font-weight: 900;
        line-height: 1.15;
        margin-bottom: 14px;
    }

    .hero-text {
        color: white;
        font-size: 15px;
        line-height: 1.8;
        margin-bottom: 25px;
    }

    .control-title {
        color: white;
        font-size: 28px;
        font-weight: 900;
        margin-top: 25px;
        margin-bottom: 14px;
    }

    .stButton > button {
        width: 100%;
        min-height: 45px;
        border-radius: 8px;
        background:
            linear-gradient(
                90deg,
                #13a89e,
                #21c7b8
            );
        color: white;
        border: 1px solid #63fff0;
        font-weight: 800;
    }

    .stButton > button:hover {
        color: white;
        border-color: white;
    }

    .result-box {
        border: 1px solid #22e8e0;
        border-radius: 15px;
        padding: 22px;
        margin-top: 28px;
        text-align: center;
        background:
            linear-gradient(
                135deg,
                rgba(10, 110, 120, 0.30),
                rgba(40, 80, 140, 0.40)
            );
    }

    .result-label {
        color: #42eee4;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 3px;
    }

    .result-letter {
        color: #45eee2;
        font-size: 50px;
        font-weight: 900;
        margin: 8px 0;
    }

    .result-confidence {
        color: white;
        font-size: 13px;
        font-weight: 700;
    }

    .footer-final {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #35e8ff;
        font-size: 11px;
        line-height: 1.9;
        border-top:
            1px solid
            rgba(255, 255, 255, 0.12);
    }

    h1, h2, h3, p, label {
        color: white !important;
    }

    @media (max-width: 768px) {

        .hero-title {
            font-size: 30px;
        }

        .control-title {
            font-size: 23px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. LOAD MODEL + LABEL ENCODER
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

    model_error = None

except Exception as e:

    model = None

    model_error = str(e)


# ============================================================
# 7. SESSION STATE
# ============================================================

if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "confidence" not in st.session_state:

    st.session_state.confidence = None


if "model_input" not in st.session_state:

    st.session_state.model_input = None


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">'
        '✍️ HandWritten Recognition System'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="online-box">'
        '● SYSTEM ONLINE'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-section">'
        '🎯 Project Overview'
        '</div>'
        '<div class="sidebar-text">'
        'A machine learning based digits '
        'recognition system that identifies '
        'handwritten digits '
        'from 0 to 9 using real-time air drawing.'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-section">'
        '🧠 Algorithms Evaluated'
        '</div>'
        '<div class="sidebar-text">'
        '<b>1.</b>&nbsp; K-Nearest Neighbors<br><br>'
        '<b>2.</b>&nbsp; Naive Bayes<br><br>'
        '<b>3.</b>&nbsp; Decision Tree<br><br>'
        '<b>4.</b>&nbsp; Logistic Regression<br><br>'
        '<b>5.</b>&nbsp; Linear SVM<br><br>'
        '<b>6.</b>&nbsp; Random Forest<br><br>'
        '<b>7.</b>&nbsp; XGBoost'
        '</div>'
        '<div class="best-model-box">'
        '🏆 Final Model: Tuned XG Boost'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-section">'
        '✋ Drawing Controls'
        '</div>'
        '<div class="sidebar-text">'
        '☝️ <b>Index finger only</b><br>'
        '<span class="small-muted">'
        'Draw the digit.'
        '</span><br><br>'
        '✋ <b>Open palm</b><br>'
        '<span class="small-muted">'
        'Hold briefly to clear drawing.'
        '</span><br><br>'
        '🔴 <b>Red cursor</b><br>'
        '<span class="small-muted">'
        'Shows fingertip position.'
        '</span>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# 9. HERO
# ============================================================

st.markdown(
    '<div class="hero-title">'
    '🔢 Handwritten Digit Recognition System'
    '</div>'
    '<div class="hero-text">'
    'Draw digits from 0 to 9 in real time using your index '
    'fingertip. The air drawing is captured '
    'through the camera, transformed into a '
    '28 × 28 grayscale model input, and classified '
    'using the tuned XG Boost pipeline.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 10. MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands


# ============================================================
# 11. VIDEO PROCESSOR
# HAND DETECTION + AIR DRAWING
# ============================================================

class AirDrawingProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.lock = threading.RLock()

        self.canvas = None

        self.cursor_x = None
        self.cursor_y = None

        self.previous_x = None
        self.previous_y = None

        self.draw_frames = 0

        self.loss_frames = 0

        self.clear_frames = 0

        self.clear_requested = False


        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            model_complexity=1,

            min_detection_confidence=0.7,

            min_tracking_confidence=0.7
        )


    # --------------------------------------------------------
    # REQUEST CLEAR
    # --------------------------------------------------------

    def request_clear(self):

        with self.lock:

            self.clear_requested = True


    # --------------------------------------------------------
    # CLEAR CANVAS
    # --------------------------------------------------------

    def clear_canvas(self):

        with self.lock:

            if self.canvas is not None:

                self.canvas[:] = 0


            self.previous_x = None
            self.previous_y = None

            self.draw_frames = 0

            self.loss_frames = 0

            self.clear_frames = 0

            self.clear_requested = False


    # --------------------------------------------------------
    # GET CANVAS
    # --------------------------------------------------------

    def get_canvas_copy(self):

        with self.lock:

            if self.canvas is None:

                return None

            return self.canvas.copy()


    # --------------------------------------------------------
    # RECEIVE CAMERA FRAME
    # --------------------------------------------------------

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )


        # Mirror camera
        image = cv2.flip(
            image,
            1
        )


        height, width = image.shape[:2]


        # Create canvas
        with self.lock:

            if (
                self.canvas is None
                or
                self.canvas.shape[:2]
                !=
                (height, width)
            ):

                self.canvas = np.zeros(

                    (
                        height,
                        width,
                        3
                    ),

                    dtype=np.uint8
                )


        # External clear button
        if self.clear_requested:

            self.clear_canvas()


        # BGR -> RGB
        rgb = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB
        )


        # Hand detection
        results = self.hands.process(
            rgb
        )


        status = "SHOW YOUR HAND"

        status_color = (
            0,
            255,
            0
        )


        # ====================================================
        # HAND FOUND
        # ====================================================

        if results.multi_hand_landmarks:

            hand = (
                results.multi_hand_landmarks[0]
            )

            lm = hand.landmark


            # Index fingertip
            raw_x = int(
                lm[8].x * width
            )

            raw_y = int(
                lm[8].y * height
            )


            # Smooth cursor
            if (
                self.cursor_x is None
                or
                self.cursor_y is None
            ):

                self.cursor_x = raw_x

                self.cursor_y = raw_y

            else:

                self.cursor_x = int(

                    SMOOTHING_ALPHA
                    * raw_x

                    +

                    (
                        1.0
                        -
                        SMOOTHING_ALPHA
                    )
                    * self.cursor_x
                )


                self.cursor_y = int(

                    SMOOTHING_ALPHA
                    * raw_y

                    +

                    (
                        1.0
                        -
                        SMOOTHING_ALPHA
                    )
                    * self.cursor_y
                )


            current_point = (

                int(
                    np.clip(
                        self.cursor_x,
                        0,
                        width - 1
                    )
                ),

                int(
                    np.clip(
                        self.cursor_y,
                        0,
                        height - 1
                    )
                )
            )


            # =================================================
            # FINGER STATES
            # =================================================

            index_up = (
                lm[8].y
                <
                lm[6].y
            )

            middle_up = (
                lm[12].y
                <
                lm[10].y
            )

            ring_up = (
                lm[16].y
                <
                lm[14].y
            )

            pinky_up = (
                lm[20].y
                <
                lm[18].y
            )


            # =================================================
            # DRAW GESTURE
            # =================================================

            draw_gesture = (

                index_up

                and not middle_up

                and not ring_up

                and not pinky_up
            )


            # =================================================
            # OPEN PALM
            # =================================================

            open_palm = (

                index_up

                and middle_up

                and ring_up

                and pinky_up
            )


            # =================================================
            # OPEN PALM CLEAR
            # =================================================

            if open_palm:

                self.clear_frames += 1

                self.draw_frames = 0

                self.previous_x = None
                self.previous_y = None


                status = (
                    "HOLD OPEN PALM TO CLEAR"
                )

                status_color = (
                    0,
                    255,
                    255
                )


                if (
                    self.clear_frames
                    >=
                    CLEAR_HOLD_FRAMES
                ):

                    self.clear_canvas()

                    status = (
                        "CANVAS CLEARED"
                    )

                    status_color = (
                        0,
                        255,
                        0
                    )


            # =================================================
            # OTHER GESTURES
            # =================================================

            else:

                self.clear_frames = 0


                # =============================================
                # DRAW
                # =============================================

                if draw_gesture:

                    self.draw_frames += 1

                    self.loss_frames = 0


                    status = (
                        "DRAWING ACTIVE"
                    )

                    status_color = (
                        0,
                        255,
                        0
                    )


                    # Red fingertip cursor
                    cv2.circle(

                        image,

                        current_point,

                        9,

                        (
                            0,
                            0,
                            255
                        ),

                        -1
                    )


                    # Stability protection
                    if (
                        self.draw_frames
                        >=
                        DRAW_STABILITY_FRAMES
                    ):

                        previous_point = None


                        if (
                            self.previous_x
                            is not None

                            and

                            self.previous_y
                            is not None
                        ):

                            previous_point = (

                                self.previous_x,

                                self.previous_y
                            )


                        with self.lock:

                            if (
                                previous_point
                                is not None
                            ):

                                cv2.line(

                                    self.canvas,

                                    previous_point,

                                    current_point,

                                    (
                                        255,
                                        255,
                                        255
                                    ),

                                    DRAW_THICKNESS,

                                    cv2.LINE_AA
                                )

                            else:

                                cv2.circle(

                                    self.canvas,

                                    current_point,

                                    DRAW_THICKNESS // 2,

                                    (
                                        255,
                                        255,
                                        255
                                    ),

                                    -1,

                                    cv2.LINE_AA
                                )


                        self.previous_x = (
                            current_point[0]
                        )

                        self.previous_y = (
                            current_point[1]
                        )


                # =============================================
                # NOT DRAWING
                # =============================================

                else:

                    self.draw_frames = 0

                    self.loss_frames += 1


                    status = (
                        "INDEX UP + OTHER "
                        "FINGERS DOWN = DRAW"
                    )

                    status_color = (
                        0,
                        255,
                        255
                    )


                    if (
                        self.loss_frames
                        >
                        GESTURE_LOSS_TOLERANCE
                    ):

                        self.previous_x = None

                        self.previous_y = None


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            self.draw_frames = 0

            self.loss_frames += 1


            if (
                self.loss_frames
                >
                GESTURE_LOSS_TOLERANCE
            ):

                self.previous_x = None

                self.previous_y = None


        # ====================================================
        # DISPLAY DRAWING ON CAMERA
        # ====================================================

        with self.lock:

            current_canvas = (

                None

                if self.canvas is None

                else self.canvas.copy()
            )


        if current_canvas is not None:

            gray_canvas = cv2.cvtColor(

                current_canvas,

                cv2.COLOR_BGR2GRAY
            )


            mask = (
                gray_canvas > 0
            )


            # Cyan visible drawing
            image[mask] = (

                255,
                255,
                80
            )


        # ====================================================
        # STATUS TEXT
        # ====================================================

        cv2.putText(

            image,

            status,

            (
                25,
                45
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            status_color,

            2,

            cv2.LINE_AA
        )


        cv2.putText(

            image,

            "Index Up + Other Fingers Down = Draw",

            (
                25,
                height - 65
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                0,
                255,
                255
            ),

            2,

            cv2.LINE_AA
        )


        cv2.putText(

            image,

            "Open Palm Hold = Clear",

            (
                25,
                height - 30
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                0,
                255,
                255
            ),

            2,

            cv2.LINE_AA
        )


        return av.VideoFrame.from_ndarray(

            image,

            format="bgr24"
        )

# ============================================================
# Digit Preprocessing (28x28)
# ============================================================

def prepare_model_input(canvas):

    if canvas is None:

        return None, None

    # Convert to Grayscale
    gray = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    # Find the drawn pixels
    points = cv2.findNonZero(gray)

    if points is None:

        return None, None
    
    # ============================================================
    # Bounding Box
    # ============================================================

    x, y, w, h = cv2.boundingRect(points)

    digit = gray[
        y:y+h,
        x:x+w
    ]

    # ============================================================
    # Keep Aspect Ratio (Square Padding)
    # ============================================================

    size = max(w, h)

    square = np.zeros(
        (size, size),
        dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    
    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = digit

    # ============================================================
    # Outer Padding
    # ============================================================

    digit = cv2.copyMakeBorder(
        square,

        20,

        20,

        20,

        20,

        cv2.BORDER_CONSTANT,

        value=0)

    # ============================================================
    # Resize to 28×28
    # ============================================================

    digit = cv2.resize(
        digit,
        (MODEL_WIDTH, MODEL_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    digit = digit.astype(
        np.float32
    ) / 255.0

    # Flatten
    features = digit.reshape(
        1,
        -1
    )

    # Feature Check
    if features.shape[1] != EXPECTED_FEATURES:

        raise ValueError(

            f"Expected {EXPECTED_FEATURES} features "

            f"but got {features.shape[1]}"

        )

    return features, digit


# ============================================================
# Digit Prediction
# ============================================================

def predict_digit(canvas):

    if model is None:

        raise RuntimeError(

            model_error

            or

            "Model unavailable."

        )

    # Prepare Input
    features, processed_digit = prepare_model_input(
        canvas
    )

    if features is None:

        raise ValueError(

            "No digit detected. Draw a digit first."

        )

    # Feature Check
    if hasattr(model, "n_features_in_"):

        expected = int(model.n_features_in_)

        actual = int(features.shape[1])

        if expected != actual:

            raise ValueError(

                f"Model expects {expected} features "

                f"but received {actual}."

            )

    # Prediction
    prediction = int(

        model.predict(features)[0]

    )

    # Confidence
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            features
        )[0]

        confidence = float(

            np.max(probabilities) * 100

        )

    return (

        prediction,

        confidence,

        processed_digit

    )

# ============================================================
# 15. WEBRTC CONFIG
# ============================================================

RTC_CONFIGURATION = RTCConfiguration(

    {
        "iceServers": [

            {
                "urls": [

                    "stun:"
                    "stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# ============================================================
# 16. CAMERA
# ============================================================

camera_left, camera_center, camera_right = (
    st.columns(
        [
            0.05,
            0.90,
            0.05
        ]
    )
)


with camera_center:

    webrtc_ctx = webrtc_streamer(

        key="airdraw-camera",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration=RTC_CONFIGURATION,

        video_processor_factory=(
            AirDrawingProcessor
        ),

        media_stream_constraints={

            "video": True,

            "audio": False
        },

        async_processing=True
    )


# ============================================================
# 17. CONTROLS
# ============================================================

st.markdown(
    '<div class="control-title">'
    '🎮 Air Drawing Controls'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    clear_clicked = st.button(

        "🗑 Clear Air Drawing",

        use_container_width=True
    )


with col2:

    predict_clicked = st.button(

        "✨ Predict Air Drawing",

        use_container_width=True
    )


# ============================================================
# 18. CLEAR BUTTON
# ============================================================

if clear_clicked:

    processor = getattr(

        webrtc_ctx,

        "video_processor",

        None
    )


    if processor is None:

        st.warning(
            "Start the camera first."
        )

    else:

        processor.request_clear()


        st.session_state.prediction = None

        st.session_state.confidence = None

        st.session_state.model_input = None


        st.success(
            "Air drawing cleared!"
        )


# ============================================================
# 19. PREDICT BUTTON
# ============================================================

if predict_clicked:

    processor = getattr(

        webrtc_ctx,

        "video_processor",

        None
    )


    if processor is None:

        st.warning(

            "Start the camera and "
            "draw a digit first."
        )


    else:

        canvas = (
            processor.get_canvas_copy()
        )


        try:

            (
                prediction,

                confidence,

                model_input

            ) = predict_digit(
                canvas
            )


            st.session_state.prediction = (
                prediction
            )


            st.session_state.confidence = (
                confidence
            )


            st.session_state.model_input = (
                model_input
            )


        except Exception as e:

            st.error(
                str(e)
            )


# ============================================================
# 20. RESULT
# ============================================================

if (
    st.session_state.prediction
    is not None
):

    if (
        st.session_state.confidence
        is None
    ):

        confidence_text = (
            "Confidence unavailable"
        )

    else:

        confidence_text = (

            f"Confidence: "
            f"{st.session_state.confidence:.2f}%"
        )


    st.markdown(

        '<div class="result-box">'

        '<div class="result-label">'
        'DIGIT RECOGNITION RESULT'
        '</div>'

        f'<div class="result-letter">'
        f'{st.session_state.prediction}'
        f'</div>'

        f'<div class="result-confidence">'
        f'{confidence_text}'
        f'</div>'

        '</div>',

        unsafe_allow_html=True
    )


# ============================================================
# 21. EXACT MODEL INPUT
# ============================================================

if (
    st.session_state.model_input
    is not None
):

    st.markdown(
        "## 🔬 Exact Model Input"
    )


    st.image(

        st.session_state.model_input,

        caption=(
            "Exact 28 × 28 grayscale input "
            "sent to the trained "
            "XG Boost model"
        ),

        width=220
    )


# ============================================================
# 22. MODEL ERROR
# ============================================================

if model_error is not None:

    st.error(

        "Model loading failed: "
        f"{model_error}"
    )


# ============================================================
# 23. FOOTER
# ============================================================

st.markdown(
    '<div class="footer-final">'
    'Handwritten Digit Recognition System'
    '<br>'
    'Built with WebRTC, OpenCV, MediaPipe and XGBoost'
    '<br>'
    'Developed by Khaja Mainuddin'
    '</div>',
    unsafe_allow_html=True
)