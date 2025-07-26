import os 
import ffmpeg

input_folder = "data/raw_webm"
output_folder = "data/converted_wav"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder) :
    if filename.endswith(".webm") :
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".webm", ".wav"))
        print(f"Converting {filename}")

        try :
            (
                ffmpeg.input(input_path)
                .output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .run(overwrite_output=True)
            )
            print(f"Saved {output_path}")
        except ffmpeg.Error as e :
            print(f"Error converting {filename} : {e}")