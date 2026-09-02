from enum import Enum

class TaskType(str, Enum):
    GENERAL = "general"

    # Language / data
    CLASSIFICATION = "classification"
    EMBEDDINGS = "embeddings"

    # Vision
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    FACE_DETECTION = "face_detection"
    IMAGE_GENERATION = "image_generation"
    IMAGE_UPSCALING = "image_upscaling"

    # Audio
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"

    # Video
    VIDEO_GENERATION = "video_generation"
    VIDEO_UPSCALING = "video_upscaling"
    VIDEO_FRAME_ANALYSIS = "video_frame_analysis"