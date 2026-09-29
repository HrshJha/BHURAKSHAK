"""Set the native chart value axis to the requested 0–100 percent scale."""
from pathlib import Path
import os
import re
import sys
import zipfile

path = Path(sys.argv[1])
tmp = path.with_suffix(path.suffix + ".axis-patched")
chart_part = "ppt/slides/charts/chart1.xml"
patched = False

with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(tmp, "w") as target:
    for item in source.infolist():
        data = source.read(item.filename)
        if item.filename == chart_part:
            xml = data.decode("utf-8")
            if not re.search(r'<c:scaling>.*?<c:orientation val="minMax"\s*/>.*?</c:scaling>', xml):
                raise SystemExit("Expected value-axis scaling not found; chart package was not changed")
            if not re.search(r'<c:crossBetween val="between"\s*/>', xml):
                raise SystemExit("Expected value-axis crossing not found; chart package was not changed")
            if not re.search(r'<c:numFmt formatCode="General"\s*/>', xml):
                raise SystemExit("Expected value-axis XML not found; chart package was not changed")
            xml = re.sub(r'(<c:scaling>)(.*?)(</c:scaling>)', lambda m: m.group(1) + '<c:max val="100"/><c:min val="0"/>' + m.group(2) + m.group(3), xml, count=1)
            xml = re.sub(r'(<c:crossBetween val="between"\s*/>)', r'\1<c:majorUnit val="20"/>', xml, count=1)
            xml = re.sub(r'<c:numFmt formatCode="General"\s*/>', '<c:numFmt formatCode="0&quot;%&quot;" sourceLinked="0"/>', xml, count=1)
            data = xml.encode("utf-8")
            patched = True
        target.writestr(item, data)

if not patched:
    raise SystemExit("Native chart XML was not found")
os.replace(tmp, path)
