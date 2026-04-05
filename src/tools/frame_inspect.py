import os
from typing import Annotated as A

import numpy as np
from nano_vectordb import NanoVectorDB

import src.config.settings as config
from src.llm.openai import call_openai_model_with_tools
from src.utils.schema import doc as D
from src.video.database import convert_hhmmss_to_seconds


def frame_inspect_tool(
    database: A[NanoVectorDB, D("The database containing video metadata. Must be an instance of NanoVectorDB.")],
    question: A[str, D("The specific detailed question to ask about the video content during the specified time ranges. No need to add time ranges in the question.")],
    time_ranges_hhmmss: A[list[tuple], D("A list of tuples containing start and end times in HH:MM:SS format. If the time range is longer than 50 seconds, the function samples 50 evenly distributed frames.  Otherwise, it uses all frames within the specified range.")],
) -> str:
    """
    Crop the video frames based on the time ranges and ask the model a detailed question about the cropped video clips.
    Returns:
        str: The model's response to the question. If no relevant content is found within the time range,
             returns an error message: "Error: Cannot find corresponding result in the given time range."
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    video_length_secs = convert_hhmmss_to_seconds(database.get_additional_data()['video_length'])
    video_meta = database.get_additional_data()
    video_file_root = video_meta['video_file_root']
    fps = video_meta.get('fps', config.VIDEO_FPS)
    time_ranges_secs = []
    for time_range in time_ranges_hhmmss:
        start_secs = convert_hhmmss_to_seconds(time_range[0])
        end_secs = convert_hhmmss_to_seconds(time_range[1])
        if start_secs > video_length_secs:
            raise ValueError(f"One of start time {time_range[0]} exceeds video length {video_length_secs}")
        end_secs = min(end_secs, video_length_secs)
        time_ranges_secs.append((start_secs, end_secs))

    time_ranges_secs.sort(key=lambda x: x[0])
    total_time = sum(end - start for start, end in time_ranges_secs)
    max_timepoints = config.AOAI_TOOL_VLM_MAX_FRAME_NUM
    timepoints = []

    assert total_time > 0 and max_timepoints > 0

    offsets = np.linspace(0, total_time, num=max_timepoints, endpoint=False, dtype=float)

    prefix_len = []
    acc = 0
    for start, end in time_ranges_secs:
        seg_len = end - start
        prefix_len.append((acc, start, seg_len))
        acc += seg_len

    for off in offsets:
        for base, seg_start, seg_len in prefix_len:
            if off < base + seg_len:
                timepoints.append(seg_start + (off - base))
                break

    max_frame_idx = int(video_length_secs * fps) - 1
    framepoints = [
        min(max(int(round(ts * fps)), 0), max_frame_idx)
        for ts in timepoints
    ]
    framepoints = sorted(set(framepoints))[:max_timepoints]

    input_msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant to answer questions."
        },
        {
            "role": "user",
            "content": "Carefully watch the video frames. Pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons.\n\nBased on your observations, if you find content that can answer the question. If no relevant content is found within the given time range, return: `Error: Cannot find corresponding result in the given time range.`. \nQuestion: {question}\n",
        },
    ]
    input_msgs[1]['content'] = input_msgs[1]['content'].format(question=question)

    files = [
        os.path.join(video_file_root, "frames", f"frame_n{fn:06d}.jpg") for fn in framepoints
    ]
    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        image_paths=files,
        temperature=0,
        max_tokens=512,
    )
    if msgs is None:
        raise ValueError("No response from the model")
    return msgs["content"]
