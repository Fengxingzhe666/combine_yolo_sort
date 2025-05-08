from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


def add_hard_subtitle(input_video_path, output_video_path, subtitle_text, font_size):
    # 加载视频文件
    video = VideoFileClip(input_video_path)

    # 创建文本剪辑，位于视频下方居中
    subtitle = TextClip(
        subtitle_text,
        fontsize=font_size,
        font="Arial",
        color="white",
        stroke_color="black",
        stroke_width=2
    )
    subtitle = subtitle.set_position(("center", video.h - 50)).set_duration(video.duration)

    # 合成视频和字幕
    result = CompositeVideoClip([video, subtitle])

    # 导出视频
    result.write_videofile(output_video_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    input_video = r"D:\\学习\\20241231周汇报会\\2024-12-28 15-28-43.mp4"
    output_video = r"D:\\学习\\20241231周汇报会\\2024-12-28 15-28-43_with_subtitle.mp4"
    subtitle_text = "ZZKNB"
    font_size = 24  # 大约等于10pt

    add_hard_subtitle(input_video, output_video, subtitle_text, font_size)
