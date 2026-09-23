from __future__ import annotations
import argparse, re, difflib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

BASE_STOPWORDS = {
    '的','了','和','是','在','我','有','就','不','人','都','一','一个','上','也','很',
    '到','说','要','去','你','会','着','没有','看','好','自己','这','那','与','及','等',
    '并','或','但','而','为','以','于','对','从','被','把','让','使','将','已','已经',
    '可以','能够','进行','通过','由于','因此','所以','但是','然而','同时','此外','其中',
    '包括','以及','对于','关于','根据','按照','随着','为了','成为','作为','我们','公司',
    '集团','企业','报告期','年度','实现','完成','增加','减少','增长','下降','万元','亿元',
    '元','个','年','月','日'
}
NEGATION_WORDS = {
    '不','没','没有','无','非','未','别','莫','勿','不曾','不会','不能','不是','不等',
    '不足','不具','不加','不予','不无','不失','不禁','缺乏','缺少','难以','无法','未能',
    '尚未','从未','绝不','并非'
}
FINANCE_WORDS = [
    '资产负债率','流动比率','速动比率','净资产收益率','营业收入','净利润','现金流',
    '管理层讨论与分析','经营活动','现金流量','毛利率','净利率','应收账款','存货周转率',
    '总资产周转率','每股收益','净利润率','投资收益','公允价值','关联交易','商誉',
    '减值准备','营业利润','利润总额','所得税','财务费用','管理费用','销售费用','研发投入',
    '非经常性损益','扣除非经常性损益','控股股东','实际控制人','董事会','监事会','股东大会',
    '内部控制','审计意见','保留意见','无保留意见','强调事项','持续经营','重大不确定性'
]
DLUT_NEG = NEGATION_WORDS

def parse_args():
    p = argparse.ArgumentParser(description="Process MD&A and construct 26 text features.")
    p.add_argument("--mda-dir", type=Path, default=ROOT/"data/raw/mda")
    p.add_argument("--dlut-dictionary", type=Path, default=ROOT/"data/raw/dlut/dlut_sentiment.xlsx")
    p.add_argument("--custom-dict", type=Path, default=None,
                   help="Optional jieba user dictionary; the original experiment also used an embedded finance vocabulary.")
    p.add_argument("--stopwords", type=Path, default=None,
                   help="Optional stopword file; when omitted, the embedded stopword list is used.")
    p.add_argument("--output-dir", type=Path, default=ROOT/"data/processed")
    return p.parse_args()

def clean_text(text):
    text = re.sub(r"<[^>]+>","",str(text))
    return re.sub(r"\s+"," ",text).strip()

def tokenize(text, stopwords):
    import jieba
    words = jieba.lcut(text)
    return [
        w.strip() for w in words
        if w.strip() and w.strip() not in stopwords
        and re.match(r"^[\u4e00-\u9fa5a-zA-Z0-9]+$", w.strip())
    ]

def load_mda_raw(folder):
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No MD&A xlsx files in {folder}")
    expected = [
        'Symbol','ShortName','Enddate','IndustryName1','ManaDiscAnal',
        'TextualSimilarity','PositiveVocabularyNum','NegativeVocabularyNum',
        'TotalWordsNum','SentencesNum','WordsNum','EmotionTone1','EmotionTone2'
    ]
    frames = []
    for path in files:
        raw = pd.read_excel(path)
        if len(raw) >= 3:
            raw = raw.iloc[2:].reset_index(drop=True)
        for c in expected:
            if c not in raw.columns:
                raw[c] = pd.NA
        raw = raw[expected].dropna(subset=['Symbol','Enddate']).copy()
        raw['Enddate'] = pd.to_datetime(raw['Enddate'], errors='coerce')
        raw = raw.dropna(subset=['Enddate'])
        raw = raw[(raw['Enddate'].dt.month==12) & (raw['Enddate'].dt.day==31)]
        raw['year'] = raw['Enddate'].dt.year.astype(int)
        for c in [
            'TextualSimilarity','PositiveVocabularyNum','NegativeVocabularyNum',
            'TotalWordsNum','SentencesNum','WordsNum','EmotionTone1','EmotionTone2'
        ]:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
        # The active experiment filled missing EmotionTone2 values from the
        # positive/negative vocabulary counts supplied by CSMAR.
        denom = raw['PositiveVocabularyNum'] + raw['NegativeVocabularyNum']
        derived_tone2 = (raw['PositiveVocabularyNum'] - raw['NegativeVocabularyNum']) / denom.replace(0, np.nan)
        raw['EmotionTone2'] = raw['EmotionTone2'].fillna(derived_tone2)
        frames.append(raw)
    mda = pd.concat(frames, ignore_index=True)
    mda = mda.drop_duplicates(['Symbol','year'], keep='first')
    mda['Stkcd'] = mda['Symbol'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(6)
    return mda

def build_stage1(mda, custom_dict=None, stopword_file=None):
    stopwords = set(BASE_STOPWORDS)
    if stopword_file and stopword_file.exists():
        stopwords.update(
            line.strip() for line in stopword_file.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    import jieba
    for word in FINANCE_WORDS:
        jieba.add_word(word)
    if custom_dict and custom_dict.exists():
        jieba.load_userdict(str(custom_dict))

    rows=[]
    for _, row in mda.iterrows():
        text = clean_text(row['ManaDiscAnal'])
        if len(text) < 50:
            continue
        words = tokenize(text, stopwords)
        n_words = len(words)
        if n_words == 0:
            rows.append({'Stkcd':row['Stkcd'],'year':row['year'],'CleanText':''})
            continue
        sents = re.split(r'[。！？；\n\r]|(?<=[.!?])\s+', text)
        sents = [s.strip() for s in sents if len(s.strip()) > 0]
        sents = [s for s in sents if len(s) >= 5]
        sent_lens=[len(s) for s in sents]
        complex_ratio=sum(len(w)>=4 for w in words)/n_words
        digit_density=sum(c.isdigit() for c in text)/len(text)
        punc_chars='，。！？；：、（）【】《》“”‘’—…·,.!?;:()[]<>"\''
        punc_density=sum(c in punc_chars for c in text)/len(text)
        rows.append({
            'Stkcd':row['Stkcd'],'year':row['year'],
            'SentLenAvg':float(np.mean(sent_lens)) if sent_lens else np.nan,
            'SentLenStd':float(np.std(sent_lens)) if len(sent_lens)>1 else 0.0,
            'ComplexWordRatio':complex_ratio,'DigitDensity':digit_density,
            'PuncDensity':punc_density,'TTR':len(set(words))/n_words,
            'CleanText':' '.join(words)
        })
    feat=pd.DataFrame(rows)
    out=mda.merge(feat,on=['Stkcd','year'],how='left')
    out['PosRatio']=out['PositiveVocabularyNum']/out['TotalWordsNum']
    out['NegRatio']=out['NegativeVocabularyNum']/out['TotalWordsNum']

    out=out.sort_values(['Stkcd','year']).reset_index(drop=True)
    jaccard=[np.nan]*len(out); edit=[np.nan]*len(out); tfidf=[np.nan]*len(out)
    for _, group in out.groupby('Stkcd'):
        g=group.sort_values('year').reset_index()
        texts=g['CleanText'].fillna('').tolist()
        years=g['year'].tolist()
        idxs=g['index'].tolist()
        for i in range(1,len(texts)):
            if years[i]-years[i-1]!=1 or not texts[i-1] or not texts[i]:
                continue
            a=set(texts[i-1].split()); b=set(texts[i].split())
            if a|b: jaccard[idxs[i]]=len(a&b)/len(a|b)
            edit[idxs[i]]=difflib.SequenceMatcher(None,texts[i-1][:5000],texts[i][:5000]).ratio()
        if len(texts)>=2:
            try:
                mat=TfidfVectorizer(max_features=5000,min_df=1).fit_transform(texts)
                for i in range(1,len(texts)):
                    if years[i]-years[i-1]==1:
                        tfidf[idxs[i]]=cosine_similarity(mat[i-1],mat[i])[0,0]
            except Exception:
                pass
    out['Jaccard_prev']=jaccard
    out['EditSim_prev']=edit
    out['TFIDF_Cosine_prev']=tfidf

    # Fixed version used in the experiment: sentence lengths and edit similarity
    # are winsorized at the 1st/99th percentiles. Bounds are computed on the
    # complete MDA feature table before the one-year lag is applied.
    for col in ['SentLenAvg', 'SentLenStd', 'EditSim_prev']:
        if col in out.columns and out[col].notna().any():
            lo, hi = out[col].quantile(0.01), out[col].quantile(0.99)
            out[col] = out[col].clip(lo, hi)
    return out

def add_dlut(stage1, dictionary_path):
    dlut=pd.read_excel(dictionary_path)
    required=['词语','情感分类','强度','极性']
    missing=set(required)-set(dlut.columns)
    if missing:
        raise ValueError(f"DLUT dictionary missing columns: {sorted(missing)}")
    dlut=dlut[required].dropna(subset=['词语','极性']).copy()
    dlut['词语']=dlut['词语'].astype(str).str.strip()
    dlut['强度']=pd.to_numeric(dlut['强度'],errors='coerce').fillna(1)
    dlut['极性']=pd.to_numeric(dlut['极性'],errors='coerce')
    word_info={}
    for _,r in dlut.iterrows():
        if r['词语'] not in word_info:
            word_info[r['词语']] = (r['极性'],r['强度'],r['情感分类'])
    pos={w for w,(p,_,_) in word_info.items() if p==1}
    neg={w for w,(p,_,_) in word_info.items() if p==2}
    intensity={w:v[1] for w,v in word_info.items()}

    m=stage1.copy()
    pos_num=np.zeros(len(m)); neg_num=np.zeros(len(m))
    pos_int=np.full(len(m),np.nan); neg_int=np.full(len(m),np.nan)
    emo_score=np.full(len(m),np.nan); emo_tone=np.full(len(m),np.nan)
    neg_after=np.full(len(m),np.nan); pos_after=np.full(len(m),np.nan)
    for idx,text in enumerate(m['CleanText'].fillna('').values):
        words=text.split() if isinstance(text,str) else []
        if not words:
            pos_num[idx]=neg_num[idx]=0
            continue
        pints=[]; nints=[]; neg_after_count=0; pos_after_count=0
        for i,w in enumerate(words):
            is_pos=w in pos; is_neg=w in neg
            if not (is_pos or is_neg): continue
            is_negated=any(x in DLUT_NEG for x in words[max(0,i-3):i])
            value=intensity.get(w,1)
            if is_pos:
                (nints if is_negated else pints).append(value)
                neg_after_count += int(is_negated)
            else:
                (pints if is_negated else nints).append(value)
                pos_after_count += int(is_negated)
        pos_num[idx]=len(pints); neg_num[idx]=len(nints)
        if pints: pos_int[idx]=float(np.mean(pints))
        if nints: neg_int[idx]=float(np.mean(nints))
        pw=sum(pints); nw=sum(nints)
        emo_score[idx]=(pw-nw)/len(words)
        if pos_num[idx]+neg_num[idx] > 0:
            emo_tone[idx]=(pos_num[idx]-neg_num[idx])/(pos_num[idx]+neg_num[idx])
        neg_after[idx]=neg_after_count/len(words)
        pos_after[idx]=pos_after_count/len(words)
    m['DLUT_PosNum']=pos_num
    m['DLUT_NegNum']=neg_num
    m['DLUT_PosRatio']=m['DLUT_PosNum']/m['TotalWordsNum']
    m['DLUT_NegRatio']=m['DLUT_NegNum']/m['TotalWordsNum']
    m['DLUT_PosIntensity']=pos_int
    m['DLUT_NegIntensity']=neg_int
    m['DLUT_EmotionScore']=emo_score
    m['DLUT_EmotionTone']=emo_tone
    m['DLUT_NegAfterNeg']=neg_after
    m['DLUT_PosAfterNeg']=pos_after
    return m

def main():
    args=parse_args()
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    mda=load_mda_raw(args.mda_dir)
    mda.to_pickle(out/'mda_clean.pkl')
    stage1=build_stage1(mda, args.custom_dict, args.stopwords)
    stage1.to_pickle(out/'mda_features_stage1.pkl')
    stage1.drop(columns=['ManaDiscAnal','CleanText'], errors='ignore').to_pickle(out/'mda_features_stage1_notext.pkl')
    stage2=add_dlut(stage1,args.dlut_dictionary)

    stage2.drop(columns=['ManaDiscAnal','CleanText'],errors='ignore').to_pickle(out/'mda_features_stage2_notext.pkl')
    stage2.to_pickle(out/'mda_features_stage2.pkl')

    # Merge as t-1 predictors.
    model_base=pd.read_csv(out/'model_dataset_base.csv',encoding='utf-8-sig',low_memory=False)
    mda_features=[
        'TextualSimilarity','PositiveVocabularyNum','NegativeVocabularyNum',
        'EmotionTone1','EmotionTone2','PosRatio','NegRatio','SentLenAvg','SentLenStd',
        'ComplexWordRatio','DigitDensity','PuncDensity','TTR','Jaccard_prev',
        'EditSim_prev','TFIDF_Cosine_prev','DLUT_PosNum','DLUT_NegNum','DLUT_PosRatio',
        'DLUT_NegRatio','DLUT_PosIntensity','DLUT_NegIntensity','DLUT_EmotionScore',
        'DLUT_EmotionTone','DLUT_NegAfterNeg','DLUT_PosAfterNeg'
    ]
    mda_lag=stage2[['Stkcd','year']+[c for c in mda_features if c in stage2.columns]].copy()
    mda_lag['year']=mda_lag['year']+1
    model_full=model_base.merge(mda_lag,on=['Stkcd','year'],how='left')

    model_full.to_csv(out/'model_dataset_full.csv',index=False,encoding='utf-8-sig')
    print(f"MD&A rows: {len(stage2)}")
    print(f"Full model rows: {len(model_full)}")
    print(f"MD&A coverage: {model_full[mda_features].notna().any(axis=1).mean():.4f}")

if __name__ == "__main__":
    main()
