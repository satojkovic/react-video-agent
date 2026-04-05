import os
import argparse
import src.config.settings as config
from src.config.logging import logger
from src.react.agent import DVDCoreAgent
from src.video.utils import decode_video_to_frames
from src.llm.openai import extract_answer

from src.video.caption import process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DVDCoreAgent on a video.")
    parser.add_argument("video_path", type=str, help="Path of the video to process")
    parser.add_argument("question", type=str, help="Question to ask about the video")
    args = parser.parse_args()
    video_path = args.video_path
    video_id = os.path.basename(video_path).split(".")[0]

    video_db_path = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "database.json")
    captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "captions")
    frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id, "frames")

    # Decode video to frames
    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        logger.info(f"Decoding video to frames in {frames_dir}...")
        decode_video_to_frames(video_path)
        logger.info("Video decoded.")
    else:
        logger.info(f"Frames already exist in {frames_dir}.")

    # Get captions
    caption_file = os.path.join(captions_dir, "captions.json")
    if not os.path.exists(caption_file):
        logger.info("Processing video to get captions...")
        process_video(frames_dir, captions_dir)
        logger.info("Captions generated.")
    else:
        logger.info(f"Captions already exist at {caption_file}.")

    # Run the agent
    logger.info(f"Starting agent for video: {video_id}")
    agent = DVDCoreAgent(
        video_db_path=video_db_path,
        video_caption_path=caption_file,
        max_iterations=15,
    )
    answer = agent.run(args.question)
    result = extract_answer(answer[-1])
    logger.info(f"Answer: {result}")
    print(result)
