import json, pathlib
from map_reduce_refine import run_pipeline

if __name__ == "__main__":
    transcript_path = pathlib.Path("data/sample_transcript.txt")
    text = transcript_path.read_text()
    report = run_pipeline(call_id="demo_call_001", transcript=text)
    print(json.dumps(report, indent=2))