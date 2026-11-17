"""Typeset the version 3 manual from its Czech Markdown source (Windows, ReportLab)."""
from pathlib import Path
import re
import textwrap
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,PageBreak,Preformatted

root=Path(__file__).resolve().parents[1]
for family,file in (("Manual","arial.ttf"),("ManualBold","arialbd.ttf"),("Mono","consola.ttf")):
    pdfmetrics.registerFont(TTFont(family,str(Path("C:/Windows/Fonts")/file)))
ink=colors.HexColor("#173348")
teal=colors.HexColor("#08776D")
body=ParagraphStyle("body",fontName="Manual",fontSize=10,leading=15.3,spaceAfter=10,textColor=ink)
heading=ParagraphStyle("heading",fontName="ManualBold",fontSize=22,leading=27,spaceAfter=20,textColor=ink)
subtitle=ParagraphStyle("subtitle",fontName="Manual",fontSize=12,leading=18,spaceAfter=20,textColor=teal)
code=ParagraphStyle("code",fontName="Mono",fontSize=8.1,leading=12,spaceBefore=4,spaceAfter=14,backColor=colors.HexColor("#EFF5F4"),borderPadding=6)
out=root/"docs/DNForecast_Manual_cs.pdf"
doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=49,rightMargin=49,topMargin=62,bottomMargin=55,
                      title="DNForecast 3.2 | Technický manuál",author="Remind / Tomáš Bělák")
story=[]
source=(root/"docs/DNForecast.cs.md").read_text(encoding="utf-8")
intro,*chapters=re.split(r"(?m)^## ",source)
for i,chapter in enumerate(chapters):
    if i:story.append(PageBreak())
    title,text=chapter.split("\n",1)
    if not i:
        story.append(Paragraph("DNForecast 3.2",heading))
        story.append(Paragraph(escape(intro.splitlines()[1]),subtitle))
    story.append(Paragraph(escape(title),heading))
    for j,part in enumerate(text.strip().split("```")):
        if j%2:
            lines=[]
            for line in part.strip("\n").splitlines():
                lines+=textwrap.wrap(line,93,subsequent_indent="  ",replace_whitespace=False) or [""]
            story.append(Preformatted("\n".join(lines),code))
        else:
            for paragraph in part.strip().split("\n\n"):
                if paragraph.strip():story.append(Paragraph(escape(paragraph.strip()),body))

def frame(c,d):
    c.saveState()
    c.setFont("ManualBold",9);c.setFillColor(teal)
    c.drawString(49,A4[1]-34,"REMIND   /   DNFORECAST")
    c.setStrokeColor(teal);c.line(49,A4[1]-42,A4[0]-49,A4[1]-42)
    c.setFont("Manual",8);c.setFillColor(ink)
    c.drawString(49,30,"Tomáš Bělák  •  MT5 a samostatné aplikace  /  3.2")
    c.drawRightString(A4[0]-49,30,str(d.page));c.restoreState()

doc.build(story,onFirstPage=frame,onLaterPages=frame)
print(out)
