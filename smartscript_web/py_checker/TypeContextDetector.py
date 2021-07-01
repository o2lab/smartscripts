import sys
import ast
# import astunparse
from collections import deque
from pprint import pprint
import os
import pickle
from . import tokenization
from . ContextModel import *
import time
# from sklearn import metrics
import warnings

# from sklearn.exceptions import DataConversionWarning,UndefinedMetricWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)



notEnding = [',','\\']
spmPath = "/home/smartscript/smartscript_web/py_checker/ContextModel/"
Type500 = "/home/smartscript/smartscript_web/py_checker/ContextModel/"
model_folder = "/home/smartscript/smartscript_web/py_checker/ContextModel/model_type/"
max_tensor_length = 1000
pect = 1
MAX_LEN = 512  # Margin

with open('/home/smartscript/smartscript_web/py_checker/ContextModel/res.pkl','rb') as f:
    methodDict = pickle.load(f)
    
def geneDict(before, curline, after, name):
    tmp = dict()
    tmp['before'] = before
    tmp['curline'] = curline
    tmp['after'] = after
    tmp['name'] = name
    return tmp


def process(name,sourcecode,posiSt,posiEd):
    st = posiSt
    ed = posiEd
    before = sourcecode[:st]
    before = before.strip()
    line = sourcecode[st:]
    cur = 0
    curline = line.split('\n')[cur]
    while len(curline.strip())>0 and curline.strip()[-1] in notEnding:
        cur+=1
        curline = curline+'\n'+line.split('\n')[cur]
    curline = curline.strip()
    after = "\n".join(line.split('\n')[cur+1:])
    after = after.strip()
    before = before[-min(MAX_LEN,int(len(before)*pect)):]
    after = after[:min(MAX_LEN,int(len(after)*pect))]
    data = geneDict(before, curline, after, name)
    return data


# -----------------------------------------------------------------------
sp = tokenization.load_model(spmPath+'spm.model')
        
with open (Type500+'Types500.pkl', 'rb') as fp:
    Types50 = pickle.load(fp)
dile = tokenization.encode(sp,'\n####\n')

def getEmb(dt):
    
    before = dt['before']
    curline = dt['curline']
    after = dt['after']
    Name = dt['name']
    token_ids = tokenization.encode(sp, Name)

    before_emb = tokenization.encode(sp, before)
    curline_emb = tokenization.encode(sp, curline)
    after_emb = tokenization.encode(sp, after)
    
    embd = before_emb+dile+curline_emb+dile+after_emb+dile+token_ids
    if len(embd) > max_tensor_length:
        return None
    return embd

# -----------------------------------------------------------------------
class FuncCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self._name = deque()

    @property
    def name(self):
        return '.'.join(self._name)

    @name.deleter
    def name(self):
        self._name.clear()

    def visit_Name(self, node):
        self._name.appendleft(node.id)

    def visit_Attribute(self, node):
        try:
            self._name.appendleft(node.attr)
            self._name.appendleft(node.value.id)
        except AttributeError:
            self.generic_visit(node)

def getMethodCall(root):
    methodCalls = []
    for node in ast.walk(root):
        if isinstance(node, ast.Call):
            callvisitor = FuncCallVisitor()
            callvisitor.visit(node.func)
            name = callvisitor.name
            if name.startswith('self.'):
                name = name[5:]
            if '.' in name:
                var = name.split('.')[0]
                method = name.split('.')[1]
                methodCalls.append([name, var, method, node.lineno,node.col_offset])
    return methodCalls



def getTypeInfer(root,code):
    allData = []
    # names = sorted({(node.id,node.lineno,node.col_offset) for node in ast.walk(root) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)})
    names = sorted({(node.id,node.lineno,node.col_offset) for node in ast.walk(root) if isinstance(node, ast.Name) and node.id!='self'})

    # names = sorted({(node.id,node.lineno,node.col_offset) for node in ast.walk(root) if isinstance(node, ast.Name)})
    # names3 = sorted({(node.name,node.lineno,node.col_offset) for node in ast.walk(root) if isinstance(node, ast.FunctionDef)})
    # namesAll = list(set(names+names3))
    # names2 = sorted({(node.attr,node.lineno,node.col_offset) for node in ast.walk(root) if isinstance(node, ast.Attribute)})
    # names3 = sorted({(node.name,node.lineno,node.col_offset) for node in ast.walk(root) if isinstance(node, ast.FunctionDef)})
    badLen = 0
    for name in names:
        lineno = name[1]
        offset = name[2]
        varName = name[0]
        varlen = len(varName)

        line = code.split('\n')[lineno-1]
        varName = line[offset:offset+varlen]
        st = len("\n".join(code.split('\n')[:lineno-1]))+offset  
        ed = st+varlen+1
        # print(varName,code[st:st+varlen+1])
        data = process(name[0],code,st,ed)
        # print(data)
        # os._exit(0)
        embd = getEmb(data)
        if embd is None:
            badLen = badLen + 1
            continue
        allData.append([varName,lineno,offset,st,ed,embd])
    print("Total data omitted for len:",badLen)
    return allData



device = torch.device('cpu')
model_path = os.path.join(model_folder, "TypeModel.ckpt")
checkpoint = torch.load(model_path,map_location=device) ##################

detector = BugDetector(checkpoint['config']['vocab_size'], checkpoint['config']['max_src_seq_len'], model_size,
                           checkpoint['config']['dropout'])
detector.load_state_dict(checkpoint['model'])
detector.to(device)
detector.eval()

def predict(wanted):
    test_samples = wanted
    fake_lables = []
    # tokens = wanted
#         token_ids = []
#         for token in tokens:
#             token_ids.append(word2index.get(token, word2index['__UNK_TOKEN__']))
    # test_samples.append(tokens)
    for t in range(len(wanted)):
        fake_lables.append(t)
    test_samples = list(zip(test_samples, fake_lables))
    allType = []
    allPoss = []
    data_loader = torch.utils.data.DataLoader(
        test_samples,
        num_workers=0,
        batch_size=64,#len(test_samples),
        collate_fn=collate_fn,
        shuffle=False)
    for batch in tqdm(data_loader):
        seqs, seqs_lens, indices = map(lambda x: x.to(device), batch)
        # print(indices)
        

        # detector = BugDetector(checkpoint['config']['vocab_size'], checkpoint['config']['max_src_seq_len'], model_size,
        #                    checkpoint['config']['dropout'])
        # detector.load_state_dict(checkpoint['model'])
        # detector.to(device)
        # detector.eval()

        pred = detector(seqs, seqs_lens)
        pred2 = F.softmax(pred,dim=1)
        # print(str(pred2.max().data)) # tensor(0.8464)
        # poss = str(pred2.max().data)[7:-1]   # tensor(0.8464) -> 0.8464
        # print(pred2[0])
        # a = 1/0
        datadict = {}
        pred = pred.max(dim=1)[1]

        for i,po,pr in zip(indices,pred2,pred):
            datadict[i] = [po,pr]
        for i in sorted(datadict):
            po,pr = datadict[i]
            allPoss.append(str(po.max().data)[7:-1])
            allType.append(str(Types50[pr]))
        # for po in pred2:
        #     allPoss.append(str(po.max().data)[7:-1])
        
        # print("len(pred)",len(pred))
        # print("len(allPoss)",len(allPoss))
        # for pr in pred:
        #     allType.append(str(Types50[pr]))
            
    return allType, allPoss

# def predict0(wanted):
#     device = torch.device('cpu')
#     model_path = os.path.join(model_folder, "TypeModel.ckpt")
#     checkpoint = torch.load(model_path,map_location=device) ##################
#     test_samples = []
#     fake_lables = []
#     tokens = wanted
# #         token_ids = []
# #         for token in tokens:
# #             token_ids.append(word2index.get(token, word2index['__UNK_TOKEN__']))
#     test_samples.append(tokens)
#     fake_lables.append(0)
#     test_samples = list(zip(test_samples, fake_lables))
#     data_loader = torch.utils.data.DataLoader(
#         test_samples,
#         num_workers=0,
#         batch_size=1,#len(test_samples),
#         collate_fn=collate_fn,
#         shuffle=False)
#     # for batch in tqdm(
#     #         data_loader, mininterval=2, desc=' ---Predicting--- ',
#     #         leave=False):
#     for batch in data_loader:
#         seqs, seqs_lens, indices = map(lambda x: x.to(device), batch)
#         detector = BugDetector(checkpoint['config']['vocab_size'], checkpoint['config']['max_src_seq_len'], model_size,
#                            checkpoint['config']['dropout'])
#         detector.load_state_dict(checkpoint['model'])
#         detector.to(device)
#         detector.eval()
#         pred = detector(seqs, seqs_lens)
#         pred2 = F.softmax(pred,dim=1)

#         # print(str(pred2.max().data)) # tensor(0.8464)

#         poss = str(pred2.max().data)[7:-1]   # tensor(0.8464) -> 0.8464
#         pred = pred.max(dim=1)[1]
#         return str(Types50[pred]),poss


def getPredict(allData):
    results = []
    
    allEmbd = []
    for varName,lineno,offset,st,ed,embd in allData:
        allEmbd.append(embd)
    print("allEmbd",len(allEmbd))
    allType, allPoss = predict(allEmbd)
    print("allType",len(allType))
    cnt = 0
    results = []
    for varName,lineno,offset,st,ed,embd in allData:

        results.append([allType[cnt],allPoss[cnt],varName,lineno,offset,st,ed])
        cnt = cnt + 1
    
    return results
# def detect(methodCalls,inferResults):
#     print('Detecting Inconsistency...')
#     print('--------------------------------------')
#     # methodDict[type] = list(methods)
#     detectCnt = 0
#     for MCname, MCvar, MCmethod, MClineno,MCcol_offset in methodCalls:
#         for typePred,poss,varName,lineno,offset,st,ed in inferResults:
#             if MCvar == varName:# and MClineno == lineno and MCcol_offset == offset:
#                 if typePred in methodDict: # has method info
#                     if MCmethod not in methodDict[typePred]:
#                         print(MCvar,'at line',MClineno,'is predicted to be',typePred,'if defined at line',lineno,',which doesn\'t support method',MCmethod)
#                         detectCnt = detectCnt + 1
#     if detectCnt == 0:
#         print('No error found.')
#     else:
#         print('Total Error Found:',detectCnt)



def detect(methodCalls,inferResults):
    res = ""
    res += 'Detecting Inconsistency...\n' 
    res+='--------------------------------------\n'
    # methodDict[type] = list(methods)
    detectCnt = 0
    tres = dict()
    marker = dict()
    for MCname, MCvar, MCmethod, MClineno,MCcol_offset in methodCalls:
        for typePred,poss,varName,lineno,offset,st,ed in inferResults:
            if MCvar == varName:# and MClineno == lineno and MCcol_offset == offset:
                if typePred in methodDict: # has method info
                    if MCmethod not in methodDict[typePred]:
                        key = tuple([MCvar, str(MClineno), MCcol_offset, typePred,MCmethod])
                        if key not in tres:
                            tres[key] = []
                        tres[key].append(str(lineno))
                        # res+= MCvar+' at line '+str(MClineno)+' is predicted to be '+typePred+' if defined at line '+str(lineno)+', which doesn\'t support method '+MCmethod+'\n'
                        detectCnt = detectCnt + 1
    if detectCnt == 0:
        res+='No error found.\n'
        return res,[]
    else:
        detectCnt = 0
        
        for key in tres:
            MCvar, MClineno, offset, typePred,MCmethod = key
            ares= MCvar+' at line '+str(MClineno)+' is predicted to be '+typePred+' if defined at line: '
            val = tres.get(key)
            detectCnt = detectCnt + 1 
            for v in val:
                ares += str(v)+", "
            ares = ares[:-2]+', which doesn\'t support method '+MCmethod+'\n'
            res = res + ares
            key = MCvar+','+MClineno+','+str(offset)
            if key not in marker:
                marker[key] = ""
            marker[key] = marker[key]+ares
        res+='Total Error Found: '+str(detectCnt)+'\n'
    return res,marker



# def detect(methodCalls,inferResults):
#     res = ""
#     res += 'Detecting Inconsistency...\n' 
#     res+='--------------------------------------\n'
#     # methodDict[type] = list(methods)
#     detectCnt = 0
#     for MCname, MCvar, MCmethod, MClineno,MCcol_offset in methodCalls:
#         for typePred,poss,varName,lineno,offset,st,ed in inferResults:
#             if MCvar == varName:# and MClineno == lineno and MCcol_offset == offset:
#                 if typePred in methodDict: # has method info
#                     if MCmethod not in methodDict[typePred]:
#                         res+= MCvar+' at line '+str(MClineno)+' is predicted to be '+typePred+' if defined at line '+str(lineno)+', which doesn\'t support method '+MCmethod+'\n'
#                         detectCnt = detectCnt + 1
#     if detectCnt == 0:
#         res+='No error found.\n'
#     else:
#         res+='Total Error Found: '+str(detectCnt)+'\n'
#     return res

def Detector(code):
   
    root = ""
    try:
        root = ast.parse(code)
    except Exception as e:
        # print("AST ERROR: "+str(e))
        return "AST ERROR: "+str(e),'',[]
    methodCalls = getMethodCall(root)

    allmethodCalls = []
    for mc in methodCalls:
        allmethodCalls.append(tuple(mc))
    methodCalls = list(set(allmethodCalls))

    # pprint(methodCalls)

    stt = time.time()
    allData = getTypeInfer(root,code)
    inferResults = getPredict(allData)
    edd = time.time()
    time_use = ''
    if len(inferResults) !=0:
        time_use = 'Type Inference takes: '+str(round(edd-stt,3))+' s.\nFor one variable:'+ str(round((edd-stt)*1000/len(inferResults),3))+'ms.\n'
    # print('*********************************************')
    # pprint(inferResults)
    # print('*********************************************')
    inconsistencies, marker = detect(methodCalls,inferResults)

    ret = {}
    for infer in inferResults:
        if str(infer[3]) not in ret:
            ret[str(infer[3])] = []
        ret[str(infer[3])].append([infer[2],infer[0]+" - "+str(infer[1])])
    return ret,time_use+inconsistencies, marker



if __name__ == "__main__":
    Detector("")