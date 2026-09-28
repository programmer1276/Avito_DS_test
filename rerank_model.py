"""Small supervised scorer on automatically retrieved candidates.
No individual benchmark IDs are used as rules or model inputs.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from retrieval import normalize,stem_text,top_indices
from extra_retrieval import ExtraRetrieval

BASE_CONFIG=dict(title=.5,body=.35,char=.15,filters=.1,geo_floor=.2,category_floor=.3,popularity=0)

class CandidateFeatures:
    def __init__(self,base,extra,intent=None):
        self.base=base;self.extra=extra;self.intent=intent
        it=base.items
        self.titles=it.item_title_raw.fillna('').map(normalize).to_numpy()
        self.stem_titles=np.array([stem_text(t) for t in self.titles],dtype=object)
        self.meta={
            'title_chars':it.item_title_raw.fillna('').str.len().to_numpy(dtype=np.float32),
            'body_chars':np.log1p(it.item_description_raw.fillna('').str.len().to_numpy(dtype=np.float32)),
            'params_chars':np.log1p(it.item_infm_params_text.fillna('').str.len().to_numpy(dtype=np.float32)),
            'rating':pd.to_numeric(it.item_rating,errors='coerce').to_numpy(dtype=np.float32),
            'log_reviews':np.log1p(pd.to_numeric(it.item_rating_reviews_count,errors='coerce').clip(lower=0).to_numpy(dtype=np.float32)),
            'phone_hidden':it.item_is_phone_hidden.to_numpy(dtype=np.float32),
            'message_forbidden':it.item_is_message_forbidden.to_numpy(dtype=np.float32),
            'price_missing_or_negative':pd.to_numeric(it.item_price,errors='coerce').lt(0).to_numpy(dtype=np.float32),
            'log_price':np.log1p(pd.to_numeric(it.item_price,errors='coerce').clip(lower=0,upper=1e7).to_numpy(dtype=np.float32)),
        }
        self.micro=it.item_microcat_id.to_numpy()

    def retrieve(self,q):
        p=self.base.components(q);e=self.extra.components(q)
        baseline=self.base.score(p,BASE_CONFIG)
        src=self.extra.sources(p,e,BASE_CONFIG,baseline)
        # Complementary channels are selected using query/document features only.
        quotas={'baseline':600,'coverage_body':250,'word_title':150,'geo_050':150}
        idx=np.unique(np.concatenate([top_indices(src[name],k) for name,k in quotas.items()]))
        return idx,p,e,src

    def build(self,q,idx,p,e,src):
        values={}
        for name,arr in {**p,**e}.items():
            if name!='popularity':values[name]=arr[idx]
        for name in ['baseline','word_title','coverage_body','coverage_both','lexical','geo_005','geo_050']:
            arr=src[name][idx]
            values[name+'_score']=arr
            values[name+'_rank']=np.log1p(rankdata(-arr,method='min')).astype(np.float32)
            values[name+'_relative']=arr/max(float(arr.max()),1e-6)
        values['same_location']=(self.base.loc[idx]==q['search_location_id']).astype(np.float32)
        for name,arr in self.meta.items():values[name]=arr[idx]
        query=normalize(q['search_query']);st=stem_text(q['search_query']);words=st.split()
        titles=self.titles[idx];stem_titles=self.stem_titles[idx]
        values['query_in_title']=np.fromiter((query in s for s in titles),dtype=np.float32,count=len(idx))
        values['stem_query_in_title']=np.fromiter((st in s for s in stem_titles),dtype=np.float32,count=len(idx))
        values['query_chars']=np.full(len(idx),len(query),dtype=np.float32)
        values['query_words']=np.full(len(idx),len(words),dtype=np.float32)
        values['has_filters']=np.full(len(idx),bool(q['search_infm_params_text']),dtype=np.float32)
        values['category_unspecified']=np.full(len(idx),q['search_category']==0,dtype=np.float32)
        values['query_delivery']=np.full(len(idx),q['search_is_delivery_search'],dtype=np.float32)
        if self.intent is not None:
            probs=self.intent.predict_proba([q])[0]
            feat=self.intent.candidate_features(probs,self.micro[idx])
            for j,name in enumerate(['intent_prob','intent_relative','intent_log','intent_top']):values[name]=feat[:,j]
            values['intent_confidence']=np.full(len(idx),probs.max(),dtype=np.float32)
        center=self.base.centers.get(q['search_location_id'])
        has_center=bool(center and np.isfinite(center['lat']) and np.isfinite(center['lon']))
        priors=np.asarray(list(self.base.loc_prior.get(q['search_location_id'],{}).values()),dtype=float)
        norm_prior=priors/max(float(priors.sum()),1e-12)
        context=[float(has_center),float(priors.max()) if len(priors) else 0.,
                 float(np.log1p(len(priors))),float(-(norm_prior*np.log(np.maximum(norm_prior,1e-12))).sum())]
        context_names=['query_center_available','query_prior_max','query_prior_locations_log1p','query_prior_entropy']
        for name,value in zip(context_names,context):values[name]=np.full(len(idx),value,dtype=np.float32)
        self.feature_names=list(values)
        return np.column_stack(list(values.values())).astype(np.float32)
