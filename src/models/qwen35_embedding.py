import os
import torch
import torch.nn.functional as F
import unicodedata
import numpy as np
import logging

from PIL import Image
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel, Qwen3_5Model, Qwen3_5Config
from transformers import AutoTokenizer, AutoProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache

logger = logging.getLogger(__name__)

MAX_LENGTH = 8192
IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
PAD_TOKEN = "<|endoftext|>"


@dataclass
class Qwen35ForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen35ForEmbedding(Qwen3_5PreTrainedModel):
    config_class = Qwen3_5Config

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen35ForEmbeddingOutput:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return Qwen35ForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def sample_frames(frames: List[Union[str, Image.Image]], max_segments: int) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames
    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    return [frames[i] for i in frame_id_array.tolist()]


def is_image_path(path: str) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    if path.startswith(('http://', 'https://')):
        clean_path = urlparse(path).path
    else:
        clean_path = path
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    if isinstance(video, list) and len(video) > 0:
        first = video[0]
        if isinstance(first, Image.Image):
            return True
        if isinstance(first, str):
            return is_image_path(first)
    return False


class Qwen35Embedder:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        self.model = Qwen35ForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "right"

        self._has_processor = False
        try:
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.processor.tokenizer.padding_side = "right"
            self._has_processor = True
        except Exception:
            self.processor = None

        self.model.eval()

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": inputs.get("attention_mask"),
        }

    def format_model_input(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video=None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict]:
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
                instruction = instruction + "."

        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content},
        ]

        texts = [] if text is None else ([text] if isinstance(text, str) else text)
        images = [] if image is None else ([image] if not isinstance(image, list) else image)
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            videos = video

        if not texts and not images and not videos:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        for vid in videos:
            video_kwargs = {"total_pixels": self.total_pixels}
            if isinstance(vid, list):
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [("file://" + e if isinstance(e, str) else e) for e in video_content]
            elif isinstance(vid, str):
                video_content = vid if vid.startswith(("http://", "https://")) else "file://" + vid
                video_kwargs = {"fps": fps or self.fps, "max_frames": max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")
            content.append({"type": "video", "video": video_content, **video_kwargs})

        for img in images:
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(("http://", "https://")) else "file://" + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")
            content.append({"type": "image", "image": image_content, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels})

        for txt in texts:
            content.append({"type": "text", "text": txt})

        return conversation

    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        tok = self.processor.tokenizer if self._has_processor else self.tokenizer

        texts = tok.apply_chat_template(
            conversations, add_generation_prompt=False, tokenize=False
        )
        if isinstance(texts, str):
            texts = [texts]
        texts = [t.rstrip() + PAD_TOKEN for t in texts]

        has_vision = any(
            any(
                item.get("type") in ("image", "video")
                for msg in conv
                for item in (msg.get("content") if isinstance(msg.get("content"), list) else [])
            )
            for conv in conversations
        )

        if has_vision and self._has_processor:
            try:
                from qwen_vl_utils.vision_process import process_vision_info
                images, video_inputs, video_kwargs = process_vision_info(
                    conversations, return_video_metadata=True, return_video_kwargs=True
                )
            except Exception as e:
                logger.error(f"Error processing vision info: {e}")
                images, video_inputs, video_kwargs = None, None, {"do_sample_frames": False}

            if video_inputs is not None:
                videos_list, video_metadata = zip(*video_inputs)
                videos_list = list(videos_list)
                video_metadata = list(video_metadata)
            else:
                videos_list, video_metadata = None, None

            # Batched examples often have different raw resolutions; do_resize=False breaks
            # Qwen2-VL-style patch packing (invalid .view in image_processor).
            inputs = self.processor(
                text=texts, images=images, videos=videos_list, video_metadata=video_metadata,
                padding=True, do_resize=True, return_tensors="pt", **video_kwargs,
            )
        else:
            inputs = tok(
                texts, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            )
        return inputs

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped = attention_mask.flip(dims=[1])
        last_one_positions = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        conversations = [
            self.format_model_input(
                text=ele.get("text"),
                image=ele.get("image"),
                video=ele.get("video"),
                instruction=ele.get("instruction"),
                fps=ele.get("fps"),
                max_frames=ele.get("max_frames"),
            )
            for ele in inputs
        ]
        processed = self._preprocess_inputs(conversations)
        processed = {k: v.to(self.model.device) for k, v in processed.items()}
        outputs = self.forward(processed)
        embeddings = self._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
