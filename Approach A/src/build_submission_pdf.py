"""Render the submission Markdown as one A4 page; never reads or runs models."""
from pathlib import Path
import html
import re
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
from matplotlib import get_data_path


def word_count(text):
    text=re.sub(r'^\|[-:| ]+\|\s*$','',text,flags=re.M)
    return len(re.sub(r'[#*|]',' ',text).split())


def main():
    source=Path('ONE_PAGE_DESCRIPTION.md');output=Path('ONE_PAGE_DESCRIPTION.pdf')
    fonts=Path(get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('Submission',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('SubmissionBold',str(fonts/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('Submission',normal='Submission',bold='SubmissionBold')
    text=source.read_text();assert word_count(text)<=550
    body=ParagraphStyle('Body',fontName='Submission',fontSize=9.4,leading=11.5,spaceAfter=3.5,textColor=colors.HexColor('#243342'))
    title=ParagraphStyle('Title',parent=body,fontName='SubmissionBold',fontSize=13.4,leading=16.4,spaceAfter=9,textColor=colors.HexColor('#153b54'))
    heading=ParagraphStyle('Heading',parent=body,fontName='SubmissionBold',fontSize=10.2,leading=12.4,spaceBefore=3.5,spaceAfter=2,textColor=colors.HexColor('#153b54'))
    team=ParagraphStyle('Team',parent=body,fontSize=9,leading=11.5,spaceAfter=0)
    def inline(t):return re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',html.escape(t))
    story=[];blocks=text.strip().split('\n\n');width=A4[0]-76
    for block in blocks:
        if block.startswith('# '):story.append(Paragraph(inline(block[2:]),title))
        elif block.startswith('## '):story.append(Paragraph(inline(block[3:]),heading))
        elif block.startswith('**TEAM MEMBERS:') or block.startswith('**Team members:'):
            box=Table([[Paragraph(inline(block),team)]],colWidths=[width]);box.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#fff1d6')),('BOX',(0,0),(-1,-1),.7,colors.HexColor('#dc9a23')),
                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.extend([box,Spacer(1,5)])
        elif block.startswith('|'):
            rows=[]
            for line in block.splitlines():
                if re.fullmatch(r'\|[-:| ]+\|',line):continue
                rows.append([Paragraph(inline(c.strip()),body) for c in line.strip('|').split('|')])
            table=Table(rows,colWidths=[width*.74,width*.26] if len(rows[0])==2 else [width*.20,width*.17,width*.17,width*.25,width*.21]);table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef3')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#96afbd')),
                ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#ccd8df')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),1)]));story.extend([table,Spacer(1,5)])
        else:story.append(Paragraph(inline(block.replace('\n',' ')),body))
    SimpleDocTemplate(str(output),pagesize=A4,leftMargin=38,rightMargin=38,topMargin=31,bottomMargin=31,
        title='Bringing the Heat — Weather-Aware Day-Ahead Energy Forecasting and Procurement Decision Support',author='Sudip Phuyal; Manila Bhandari; Rujjal Sada; Mahsa Azarnia; Vignesh Pechiappan Ayyathurai').build(story)
    reader=PdfReader(output);assert len(reader.pages)==1,'Submission exceeds one A4 page'
    box=reader.pages[0].mediabox;assert abs(float(box.width)-A4[0])<.1 and abs(float(box.height)-A4[1])<.1
    extracted=reader.pages[0].extract_text();assert 'production trading system' in extracted
    print(f'Created {output}: one A4 page, {word_count(text)} words')

if __name__=='__main__':main()
