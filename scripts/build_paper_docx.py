"""Build an editable Word manuscript from BAO_CAO.md using Codex's Python runtime.

latex2mathml is loaded from the isolated temporary build directory. Microsoft's
installed MathML-to-OMML stylesheet supplies native, editable Word equations.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,default=Path(tempfile.gettempdir())/'nlp-scientific-paper-docx')
    args=parser.parse_args()
    build=args.build_dir
    build.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(build/'vendor'))
    from latex2mathml.converter import convert
    from lxml import etree
    from docx import Document
    from docx.shared import Cm,Pt,RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_BREAK,WD_TAB_ALIGNMENT
    from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    import latex2mathml

    transform=etree.XSLT(etree.parse(r'C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL'))
    document=Document()
    section=document.sections[0]
    section.page_width=Cm(21)
    section.page_height=Cm(29.7)
    section.top_margin=Cm(2.0)
    section.bottom_margin=Cm(2.0)
    section.left_margin=Cm(2.2)
    section.right_margin=Cm(2.2)
    section.footer_distance=Cm(.8)
    width=16.6
    normal=document.styles['Normal']
    normal.font.name='Times New Roman'
    normal.font.size=Pt(11.5)
    normal.font.color.rgb=RGBColor(0,0,0)
    normal.paragraph_format.line_spacing=1.10
    normal.paragraph_format.space_after=Pt(5)
    normal.paragraph_format.widow_control=True
    for name,size in [('Title',17),('Heading 1',13),('Heading 2',11.5)]:
        style=document.styles[name]
        style.font.name='Times New Roman'
        style.font.size=Pt(size)
        style.font.color.rgb=RGBColor(0,0,0)
        style.font.bold=True
        style.paragraph_format.space_before=Pt(11 if name!='Title' else 0)
        style.paragraph_format.space_after=Pt(5)
        style.paragraph_format.keep_with_next=True
        style.paragraph_format.keep_together=True
    document.styles['Title'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
    document.styles['Title'].paragraph_format.space_after=Pt(9)
    for style in document.styles:
        if style.type==1:
            rpr=style.element.get_or_add_rPr()
            fonts=rpr.find(qn('w:rFonts'))
            if fonts is None:
                fonts=OxmlElement('w:rFonts');rpr.append(fonts)
            for attr in ('ascii','hAnsi','eastAsia','cs'):
                fonts.set(qn('w:'+attr),'Times New Roman')
            for attr in list(fonts.attrib):
                if etree.QName(attr).localname.endswith('Theme'):del fonts.attrib[attr]
            ppr=style.element.find(qn('w:pPr'))
            if ppr is not None:
                for border in list(ppr.findall(qn('w:pBdr'))):ppr.remove(border)
    mathpr=OxmlElement('m:mathPr')
    mathfont=OxmlElement('m:mathFont');mathfont.set(qn('m:val'),'Cambria Math');mathpr.append(mathfont)
    document.settings.element.append(mathpr)
    footer=section.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    page=OxmlElement('w:fldSimple');page.set(qn('w:instr'),'PAGE')
    footer._p.append(page)

    stats=dict(display_equations=0,inline_equations=0,tables=0,hyperlinks=0,algorithm_lines=0)
    math_cache={}

    def math(tex):
        if tex not in math_cache:
            # Plain upright function labels avoid empty function arguments in
            # the Office XSL transform while preserving mathematical meaning.
            normalized=tex.replace('\\operatorname','\\mathrm')
            normalized=re.sub(r'(?<!\\)\|([^|]+?)(?<!\\)\|',r'\\left|\1\\right|',normalized)
            normalized=re.sub(r'\\\|(.+?)\\\|',r'\\left\\| \1 \\right\\|',normalized)
            normalized=normalized.replace('\\alpha/\\beta',r'\frac{\alpha}{\beta}')
            normalized=normalized.replace('(L-m)/(t-m)',r'\frac{L-m}{t-m}')
            normalized=normalized.replace('(M-L)/(M-t)',r'\frac{M-L}{M-t}')
            normalized=normalized.replace('1/r_q',r'\frac{1}{r_q}')
            source=convert(normalized)
            mathml=etree.fromstring(source.encode('utf-8'))
            mml='{http://www.w3.org/1998/Math/MathML}'
            pairs={'(':')','[':']','{':'}','|':'|','‖':'‖','∥':'∥'}
            for node in list(mathml.iter(mml+'mrow'))[::-1]:
                children=list(node)
                if not children or children[0].tag!=mml+'mo':continue
                opening=children[0].text
                if opening not in pairs:continue
                closing=children[-1].text if children[-1].tag==mml+'mo' else None
                if closing==pairs[opening]:
                    middle=children[1:-1]
                elif opening=='{' and children[-1].tag==mml+'mtable':
                    closing='';middle=children[1:]
                else:continue
                parent=node.getparent()
                if parent is not None:
                    fenced=etree.Element(mml+'mfenced',{'open':opening,'close':closing,'separators':''})
                    for child in middle:fenced.append(child)
                    parent.replace(node,fenced)
            root=transform(mathml).getroot()
            if root.tag!=qn('m:oMath'):
                raise RuntimeError('Unexpected math conversion output: '+tex)
            # The Office XSL leaves a n-ary argument empty when MathML places
            # the operand in following siblings. Associate those operands.
            for nary in root.findall('.//'+qn('m:nary')):
                argument=nary.find(qn('m:e'))
                if argument is not None and not ''.join(argument.itertext()).strip():
                    following=nary.getnext()
                    while following is not None:
                        value=''.join(following.itertext()).strip()
                        if following.tag==qn('m:r') and value in (',',';','.','=','+','−'):break
                        next_element=following.getnext()
                        argument.append(following)
                        following=next_element
            for run in root.findall('.//'+qn('m:r')):
                text=''.join(run.findall(qn('m:t'))[0].itertext()) if run.findall(qn('m:t')) else ''
                if re.fullmatch(r'[A-Za-z]{2,}',text):
                    mprops=run.find(qn('m:rPr'))
                    if mprops is None:mprops=OxmlElement('m:rPr');run.insert(0,mprops)
                    if mprops.find(qn('m:nor')) is None:mprops.append(OxmlElement('m:nor'))
                props=run.find(qn('w:rPr'))
                if props is None:
                    props=OxmlElement('w:rPr')
                    run.insert(1 if run.find(qn('m:rPr')) is not None else 0,props)
                for tag,value in [('color','000000'),('sz','23'),('szCs','23')]:
                    element=props.find(qn('w:'+tag))
                    if element is None:element=OxmlElement('w:'+tag);props.append(element)
                    element.set(qn('w:val'),value)
            for props in root.findall('.//'+qn('m:ctrlPr')):
                rpr=props.find(qn('w:rPr'))
                if rpr is None:rpr=OxmlElement('w:rPr');props.append(rpr)
                color=rpr.find(qn('w:color'))
                if color is None:color=OxmlElement('w:color');rpr.append(color)
                color.set(qn('w:val'),'000000')
            math_cache[tex]=root
        return deepcopy(math_cache[tex])

    def hyperlink(paragraph,label,target):
        if not re.match(r'^[a-zA-Z]+:',target):
            target=os.path.relpath(ROOT/target,ROOT/'docs/report').replace('\\','/')
        relation=paragraph.part.relate_to(target,RT.HYPERLINK,is_external=True)
        link=OxmlElement('w:hyperlink');link.set(qn('r:id'),relation)
        run=OxmlElement('w:r')
        props=OxmlElement('w:rPr')
        color=OxmlElement('w:color');color.set(qn('w:val'),'1F4E79');props.append(color)
        run.append(props)
        text=OxmlElement('w:t');text.text=label;run.append(text)
        link.append(run);paragraph._p.append(link)
        stats['hyperlinks']+=1

    inline_pattern=re.compile(r'(\\\(.+?\\\)|\[[^\]]+\]\([^)]+\)|\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)')

    def inline(paragraph,text):
        cursor=0
        for match in inline_pattern.finditer(text):
            paragraph.add_run(text[cursor:match.start()])
            token=match.group()
            if token.startswith('\\('):
                paragraph._p.append(math(token[2:-2]))
                stats['inline_equations']+=1
            elif token.startswith('['):
                parsed=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',token)
                hyperlink(paragraph,*parsed.groups())
            elif token.startswith('**'):
                paragraph.add_run(token[2:-2]).bold=True
            elif token.startswith('`'):
                run=paragraph.add_run(token[1:-1]);run.font.name='Consolas';run.font.size=Pt(9.5)
            else:
                paragraph.add_run(token[1:-1]).italic=True
            cursor=match.end()
        paragraph.add_run(text[cursor:])

    table_widths={
        1:[3.2,1.8,2.9,2.9,3.0],
        2:[5.0,11.6],
        3:[2.6,2.1,1.6,3.3,3.3,2.2],
        4:[2.5,3.4,4.1,3.3,3.3],
        5:[2.3,2.3,2.3,5.0,4.0],
        6:[7.0,3.5,2.7,3.4],
        7:[8.1,4.7,3.8],
        8:[2.3,1.9,3.3,3.5,2.7,2.9],
        9:[2.4,2.0,2.0,3.5,2.5,4.2],
        10:[2.4,2.0,3.0,4.9,3.5],
    }

    def add_table(lines):
        data=[[cell.strip() for cell in line.strip().strip('|').split('|')] for line in lines]
        data=[data[0]]+data[2:]
        stats['tables']+=1
        index=stats['tables']
        widths=table_widths[index]
        assert len(widths)==len(data[0])
        grid=document.add_table(rows=len(data),cols=len(widths))
        grid.alignment=WD_TABLE_ALIGNMENT.CENTER
        grid.autofit=False
        for col,centimeters in zip(grid.columns,widths):col.width=Cm(centimeters)
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            border=OxmlElement('w:'+edge)
            for key,value in [('val','single'),('sz','4'),('color','D9D9D9')]:border.set(qn('w:'+key),value)
            borders.append(border)
        grid._tbl.tblPr.append(borders)
        for ri,row in enumerate(grid.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            if ri==0:row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for ci,(cell,centimeters,value) in enumerate(zip(row.cells,widths,data[ri])):
                cell.width=Cm(centimeters)
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                props=cell._tc.get_or_add_tcPr()
                padding=OxmlElement('w:tcMar')
                for edge in ['top','bottom','left','right']:
                    margin=OxmlElement('w:'+edge);margin.set(qn('w:w'),'70');margin.set(qn('w:type'),'dxa');padding.append(margin)
                props.append(padding)
                if ri==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7EEF5');props.append(shade)
                paragraph=cell.paragraphs[0]
                paragraph.paragraph_format.space_before=Pt(2)
                paragraph.paragraph_format.space_after=Pt(2)
                paragraph.paragraph_format.line_spacing=1.0
                paragraph.paragraph_format.keep_with_next=ri==0
                paragraph.alignment=WD_ALIGN_PARAGRAPH.LEFT if index in (2,6,7) and ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                if index==2 and ci==1:paragraph.alignment=WD_ALIGN_PARAGRAPH.LEFT
                inline(paragraph,value)
                for run in paragraph.runs:
                    run.font.size=Pt(10)
                    run.bold=ri==0
                if index==5 and ri>0 and data[ri][0]=='0,5':
                    for run in paragraph.runs:run.bold=True
        gap=document.add_paragraph()
        gap.paragraph_format.space_after=Pt(1)
        gap.paragraph_format.space_before=Pt(0)
        gap.paragraph_format.line_spacing=Pt(2)
        gap.add_run().font.size=Pt(2)

    source=(ROOT/'BAO_CAO.md').read_text(encoding='utf-8')
    lines=source.splitlines()
    i=0
    in_references=False
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line=='\\[':
            block=[];i+=1
            while lines[i].strip()!='\\]':block.append(lines[i]);i+=1
            tex='\n'.join(block)
            label=re.search(r'\\tag\{(\d+)\}',tex).group(1)
            tex=re.sub(r'\\tag\{\d+\}','',tex).strip()
            # Punctuation belongs to the surrounding sentence, not math layout.
            paragraph=document.add_paragraph()
            paragraph.paragraph_format.keep_together=True
            paragraph.paragraph_format.space_before=Pt(4)
            paragraph.paragraph_format.space_after=Pt(7)
            paragraph.paragraph_format.tab_stops.add_tab_stop(Cm(width/2),WD_TAB_ALIGNMENT.CENTER)
            paragraph.paragraph_format.tab_stops.add_tab_stop(Cm(width),WD_TAB_ALIGNMENT.RIGHT)
            paragraph.add_run('\t')
            paragraph._p.append(math(tex))
            paragraph.add_run('\t('+label+')')
            stats['display_equations']+=1
        elif line.startswith('```'):
            i+=1
            while not lines[i].strip().startswith('```'):
                paragraph=document.add_paragraph()
                paragraph.paragraph_format.left_indent=Cm(.25)
                paragraph.paragraph_format.space_after=Pt(1)
                paragraph.paragraph_format.line_spacing=1.0
                paragraph.paragraph_format.keep_together=True
                paragraph.paragraph_format.keep_with_next=not lines[i+1].strip().startswith('```')
                run=paragraph.add_run(lines[i]);run.font.name='Consolas';run.font.size=Pt(9.5)
                stats['algorithm_lines']+=1
                i+=1
        elif line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].lstrip().startswith('|'):
                block.append(lines[i]);i+=1
            add_table(block)
            continue
        elif line.startswith('# '):
            paragraph=document.add_paragraph(style='Title');inline(paragraph,line[2:])
        elif line.startswith('## '):
            paragraph=document.add_paragraph(style='Heading 1');inline(paragraph,line[3:])
            if line[3:]=='Tài liệu tham khảo':in_references=True
        elif line.startswith('### '):
            paragraph=document.add_paragraph(style='Heading 2');inline(paragraph,line[4:])
        else:
            paragraph=document.add_paragraph()
            paragraph.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
            inline(paragraph,line)
            following=next((value.strip() for value in lines[i+1:] if value.strip()),'')
            if following=='\\[':
                paragraph.paragraph_format.keep_with_next=True
            if line.startswith('**Bảng ') or line.startswith('**Thuật toán '):
                paragraph.paragraph_format.keep_with_next=True
                paragraph.paragraph_format.space_before=Pt(6)
                paragraph.paragraph_format.space_after=Pt(5)
                paragraph.paragraph_format.keep_together=True
                for run in paragraph.runs:run.font.size=Pt(10.5)
            elif i<6:
                paragraph.alignment=WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_after=Pt(3)
                paragraph.paragraph_format.keep_with_next=True
            elif in_references:
                paragraph.alignment=WD_ALIGN_PARAGRAPH.LEFT
                paragraph.paragraph_format.left_indent=Cm(.65)
                paragraph.paragraph_format.first_line_indent=Cm(-.65)
                paragraph.paragraph_format.space_after=Pt(5)
                for run in paragraph.runs:run.font.size=Pt(10.5)
        i+=1

    document.core_properties.title=lines[0][2:]
    document.core_properties.author='Vương Đức Hùng'
    document.core_properties.subject='Phân đoạn ngữ nghĩa tiếng Việt cho truy xuất RAG'
    document.core_properties.comments='Bản bài báo khoa học từ BAO_CAO.md; công thức native OMML.'
    document.core_properties.created=datetime.now(timezone.utc)
    document.core_properties.modified=datetime.now(timezone.utc)
    output=build/'Bao_Cao.docx'
    document.save(output)
    assert stats['tables']==10 and stats['display_equations']==11
    audit=dict(source_markdown=str(ROOT/'BAO_CAO.md'),source_sha256=hashlib.sha256((ROOT/'BAO_CAO.md').read_bytes()).hexdigest(),
        output=str(output),python=sys.executable,latex2mathml_version=latex2mathml.__version__,
        math_transform=r'C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL',**stats)
    (build/'build_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
