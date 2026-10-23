"""Build the Czech PDF manual; requires ReportLab. Sources stay reviewable in Git."""
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
    Spacer, PageBreak, Table, TableStyle, Preformatted, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/MQL5GPULibrary_Manual_cs.pdf'
FONT = Path('C:/Windows/Fonts')
for name, filename in [('Body','arial.ttf'),('BodyBold','arialbd.ttf'),('Code','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / filename)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
NAVY=colors.HexColor('#132F47'); TEAL=colors.HexColor('#007C87'); LIGHT=colors.HexColor('#EDF4F7')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Text',fontName='Body',fontSize=9.4,leading=14,spaceAfter=8,textColor=NAVY))
styles.add(ParagraphStyle(name='TitleCS',fontName='BodyBold',fontSize=34,leading=41,textColor=NAVY,spaceAfter=18))
styles.add(ParagraphStyle(name='Chapter',fontName='BodyBold',fontSize=19,leading=24,textColor=TEAL,spaceAfter=16,keepWithNext=True))
styles.add(ParagraphStyle(name='Cell',fontName='Body',fontSize=8.2,leading=11.4,textColor=NAVY))
styles.add(ParagraphStyle(name='CodeCS',fontName='Code',fontSize=7.2,leading=10,spaceAfter=12,backColor=LIGHT,borderPadding=8))
styles.add(ParagraphStyle(name='Small',fontName='Body',fontSize=8,leading=11,textColor=NAVY,spaceAfter=6))
styles.add(ParagraphStyle(name='ApiName',fontName='BodyBold',fontSize=10.2,leading=14,textColor=TEAL,spaceBefore=8,spaceAfter=5,keepWithNext=True))

class Manual(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=53,bottomMargin=48,
            title='MQL5 GPU Library 2.3.0 — technický manuál',author='RemindCZ / Tomáš Bělák')
        self.addPageTemplates(PageTemplate(id='main',frames=[Frame(46,48,A4[0]-92,A4[1]-101,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.page))
    def page(self,canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL);canvas.setLineWidth(.7)
        canvas.line(46,A4[1]-35,A4[0]-46,A4[1]-35)
        canvas.setFont('Body',7.4);canvas.setFillColor(NAVY)
        canvas.drawString(46,A4[1]-27,'REMINDCZ  /  MQL5 GPU LIBRARY 2.3.0')
        canvas.drawString(46,29,'Technický manuál • Windows x64 • CUDA • MT5 / Python')
        canvas.drawRightString(A4[0]-46,29,str(doc.page))
        canvas.restoreState()
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='Chapter':
            label=flowable.getPlainText(); key='chapter-'+label.split('.')[0]
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(label,key,level=0,closed=False)
            self.notify('TOCEntry',(0,label,self.page,key))

def para(text,style='Text'):
    return Paragraph(escape(text),styles[style])

def diagram():
    d=Drawing(500,180)
    boxes=[(0,65,92,48,'4 vstupy','24 kroků'),(140,125,100,42,'LSTM 32','větev A'),
           (140,15,100,42,'GRU 16','větev B'),(285,65,100,48,'Dense 24','concat → tanh'),
           (420,65,80,48,'6 výstupů','lineární hlava')]
    for x,y,w,h,a,b in boxes:
        d.add(Rect(x,y,w,h,rx=5,ry=5,fillColor=LIGHT,strokeColor=TEAL))
        d.add(String(x+w/2,y+h-17,a,fontName='BodyBold',fontSize=9,textAnchor='middle',fillColor=NAVY))
        d.add(String(x+w/2,y+12,b,fontName='Body',fontSize=8,textAnchor='middle',fillColor=NAVY))
    for x1,y1,x2,y2 in [(92,95,140,145),(92,80,140,36),(240,145,285,95),(240,36,285,80),(385,89,420,89)]:
        d.add(Line(x1,y1,x2,y2,strokeColor=TEAL,strokeWidth=1.2))
    return d

API_NOTES={
'DN_GetVersion':'Vrací 20300 pro verzi 2.3.0. Umožňuje odmítnout nekompatibilní DLL před konfigurací.',
'DN_Create':'Vrací kladný handle nebo 0 při selhání vytvoření CUDA prostředků.',
'DN_Free':'Blokující uvolnění handle včetně zrušení a dokončení workeru. Opakované volání je bezpečné.',
'DN_BeginGraph':'Zahájí prázdný graf s pevným počtem vlastností a časových kroků. Vymaže předchozí model a data.',
'DN_AddGraphNode':'Vrací ID uzlu 0 nebo vyšší, -1 při chybě. Typy a aktivace viz kapitola 4. parents je pole parent_count platných int.',
'DN_SetGraphOutput':'Vybere uzel pro poslední časový krok a vytvoří lineární hlavu s outputs výstupy. Všechny uzly musejí přispívat do výstupu.',
'DN_GetInputDim':'Vrací počet vstupních vlastností, 0 při neúplné konfiguraci nebo chybě.',
'DN_GetOutputDim':'Vrací počet výstupů nakonfigurované hlavy, 0 bez hlavy nebo při chybě.',
'DN_GetSequenceLength':'Vrací délku sekvence, 0 při chybě.',
'DN_SetSequenceLength':'Nastaví kladnou délku sekvence. Změna zneplatní načtená trénovací data.',
'DN_SetMiniBatchSize':'Nastaví kladnou velikost minibatch; ověřuje se také bezpečnost součinů rozměrů.',
'DN_AddLayerEx':'Původní sekvenční API. act je 0=LSTM, 1=GRU, 2=RNN; ln musí být 0. in=0 umožňuje odvození šířky. V grafovém režimu se odmítá.',
'DN_AddGRULayer':'Zkratka sekvenčního API pro GRU; stejné rozměrové a dropout podmínky.',
'DN_AddRNNLayer':'Zkratka sekvenčního API pro tanh RNN; stejné rozměrové a dropout podmínky.',
'DN_SetGradClip':'Nastaví konečný nezáporný limit globální normy gradientu; 0 vypíná clipping.',
'DN_SetOutputDim':'Vytvoří nebo nahradí lineární hlavu za poslední vrstvou. Zneplatní načtená data; v grafu ověří spojení všech uzlů.',
'DN_LoadBatch':'Zkopíruje X a T do interního úložiště. in=sequence×features, out=šířka hlavy, l=0. Kontrolujte délky vlastních bufferů.',
'DN_PredictBatch':'Predikuje pro batch sekvencí, zapíše batch×outputs double do Y. l=0. Během tréninku téhož handle selže.',
'DN_SnapshotWeights':'Vytvoří interní snapshot vah, bez dat, normalizace a úplného optimizer/RNG stavu.',
'DN_RestoreWeights':'Obnoví úplný snapshot a resetuje Adam. Bez existujícího snapshotu selže.',
'DN_Train':'Blokující trénink. epochs>0, lr>0, target_mse a wd konečné nezáporné. Návrat 1 značí dokončení běhu.',
'DN_TrainAsync':'Spustí worker a vrátí 1 při úspěšném zahájení. Dokončení a případné chyby ověřte přes status/result.',
'DN_GetTrainingStatus':'Stavy 0/1/2/-1 jsou popsány v kapitole 7. Zastavený běh končí stavem 2.',
'DN_GetTrainingResult':'Zapíše výslednou MSE a počet dosažených epoch. Čtěte po dokončení; ukazatele musejí být platné.',
'DN_StopTraining':'Neblokující žádost o ukončení mezi minibatchi; neznamená okamžité uvolnění GPU.',
'DN_GetProgressAll':'Zápis do 4 int a 7 double ukazatelů, pořadí přesně podle deklarace. Jednotlivě atomické vzorky, nikoli společný snapshot.',
'DN_GetLayerCount':'Vrací počet vrstev/uzlů včetně existující lineární výstupní hlavy, 0 při chybě.',
'DN_GetLayerWeightNorm':'L2 norma hlavní matice W vrstvy l (od 0); index rovný počtu uzlů označuje výstupní hlavu. Nezahrnuje bias ani oddělené GRU kandidátní matice; Identity vrací 0.',
'DN_GetGradNorm':'Vrací poslední globální normu gradientu před clippingem; dostupná také v progress.',
'DN_SaveState':'Připraví text checkpointu a vrátí jeho délku v bajtech včetně NUL, 0 při chybě.',
'DN_GetState':'Zkopíruje předem připravenou serializaci do buf. max_len musí stačit včetně NUL. Vrací 1/0.',
'DN_LoadState':'Načte NUL-ukončený checkpoint. Neplatný soubor zachová model; úspěch resetuje data a optimizer.',
'DN_GetError':'Zapíše ukončenou UTF-16 zprávu do len prvků typu short; při zkrácení nejvýše len−1 znaků. Diagnostika je globální.'}
PROGRESS={'Epoch':'Aktuálně dosažená epocha','TotalEpochs':'Plánovaný počet epoch','MiniBatch':'Aktuální minibatch',
'TotalMiniBatches':'Počet minibatchí v epoše','LR':'Aktuální learning rate','MSE':'Poslední vyhodnocená trénovací MSE',
'BestMSE':'Nejlepší vyhodnocená trénovací MSE','GradNorm':'Poslední norma gradientu před clippingem',
'TotalSteps':'Počet optimizer kroků v běhu','Percent':'Průběh v procentech; dokončení včetně stop nastaví 100',
'ElapsedSec':'Uplynulá doba v sekundách','ETASec':'Odhad zbývajících sekund; po dokončení 0'}

def code_block(lines):
    # Wrap at whitespace, then a hard boundary for long DSL tokens; retain all characters.
    import textwrap
    wrapped=[]
    for line in lines:
        wrapped.extend(textwrap.wrap(line,width=103,replace_whitespace=False,drop_whitespace=False) or [''])
    return Preformatted('\n'.join(wrapped),styles['CodeCS'])

def build():
    story=[Spacer(1,45),para('MQL5 GPU Library','TitleCS'),para('Technický a uživatelský manuál • verze 2.3.0'),
        para('Grafové neuronové sítě pro finanční časové řady', 'Text'),Spacer(1,24),diagram(),Spacer(1,22),
        para('DLL pro MetaTrader 5 a samostatné aplikace ve Windows. Instalace pro IC Trading MT5, konfigurace topologie, datový kontrakt, více trhů, přímá predikce horizontů, checkpointy a úplná reference C ABI.'),
        Spacer(1,22),para('Autor projektu: Tomáš Bělák / RemindCZ'),
        para('Rozsah: podporované acyklické grafy; 45 exportů; CUDA float32; double na hranici API.'),
        para('Technické ověření a kvalita tržní predikce jsou dvě samostatná měření. Tato verze neobsahuje prokázaný výsledek na živém účtu IC Trading.','Small'),PageBreak(),
        para('Obsah','TitleCS')]
    toc=TableOfContents()
    toc.levelStyles=[ParagraphStyle(name='TOC',fontName='Body',fontSize=10,leading=17,textColor=NAVY,spaceBefore=5)]
    story.append(toc)
    lines=(ROOT/'docs/MANUAL.cs.md').read_text(encoding='utf-8').splitlines()
    i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('## '):
            story.extend([PageBreak(),para(line[3:],'Chapter')]);i+=1
        elif line.startswith('# ') or not line.strip(): i+=1
        elif line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            story.append(code_block(block));i+=1
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in cells):rows.append([para(c,'Cell') for c in cells])
                i+=1
            widths=([183,320] if len(rows[0])==2 else [150,125,228])
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LINEBELOW',(0,0),(-1,0),.8,TEAL),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#CAD8DF')),
                ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.extend([table,Spacer(1,12)])
        elif re.match(r'^\d+\. ',line):
            story.append(para(line));i+=1
        else:
            block=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','```','|')) and not re.match(r'^\d+\. ',lines[i]):
                block.append(lines[i]);i+=1
            story.append(para(' '.join(block)))
    header=(ROOT/'include/MQL5GPULibrary.h').read_text()
    declarations=re.findall(r'DN_API (.+?);',header)
    assert len(declarations)==45, 'Update manual export count after ABI changes'
    for declaration in declarations:
        declaration=declaration.replace('DN_CALL ','')+';'
        name=re.search(r'(DN_\w+)\(',declaration)[1]
        note=API_NOTES.get(name)
        if note is None:
            suffix=name.removeprefix('DN_GetProgress')
            note=PROGRESS[suffix]+'. Čtěte během tréninku; při neplatném handle se vrací 0.'
        story.append(KeepTogether([para(name,'ApiName'),code_block([declaration]),para(note,'Small')]))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    Manual(OUT).multiBuild(story)
    print(OUT)

if __name__=='__main__':build()
