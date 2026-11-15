import sys
from pathlib import Path

from error_classifier.recommend import recommend_text

sys.stdout.write(recommend_text(Path(__file__).resolve().parents[1]))
