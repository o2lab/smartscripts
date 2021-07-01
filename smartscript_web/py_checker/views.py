import os
import json
import logging
import time
import json
import git
import requests
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.decorators.csrf import csrf_exempt
from . import checker
from . import BPEModel
from . import TypeModelServer, kubeModelServer, newTypeModel, PYIModel, TypeContextDetector, BaseContextDetector

logger = logging.getLogger(__name__)
def linecharts(request):
    return render(request, 'py_checker/line-stack.html')
def api(request):
    return render(request, 'py_checker/api.html')
def bug(request):
    return render(request, 'py_checker/bug.html')
def Report(request):
    return render(request, 'py_checker/Report.html')
def echarts(request):
    return render(request, 'py_checker/echarts.html')
def pychecker(request):
    return render(request, 'py_checker/pychecker.html')
def index(request):
    return render(request, 'py_checker/index.html')
def model(request):
    return render(request, 'py_checker/model.html')
def typeChecker(request):
    return render(request, 'py_checker/typeChecker.html')
def typeGenerator(request):
    return render(request, 'py_checker/typeGenerator.html')
def smartkube(request):
    return render(request, 'py_checker/smartkube.html')

def clcheck(request):
    return render(request, 'py_checker/clcheck/index.html')

def reload(request):
    saveClient(request, "reload")
    requests.get("http://127.0.0.1:8877")
    return JsonResponse({'issues': "Reloading now. Please go to http://smart-scripts.org/ in 5 seconds."})
@csrf_exempt
def githook(request):
    repo = git.Repo('/home/smartscript/')
    for remote in repo.remotes:
        remote.fetch()
    repo.git.reset('--hard')
    repo.remote().pull()
    if request.method == "POST":
        data = request.body
        method = 'POST'
    else:
        data = request.body
        method = 'GET'
    localtime = time.asctime( time.localtime(time.time()) )
    with open("/home/smartscript/smartscript_web/py_checker/gitlog.txt",'a') as f:
        f.write("----------------------------------  GitHub Update at: ")
        f.write(str(localtime)+" using "+method)
        f.write(" with data:\n")
        f.write(str(data))
        f.write("\n")
        f.flush()
    requests.get("http://127.0.0.1:8877")
    return JsonResponse({'issues': "OK"})
def datacnt(request):
    with open("/home/smartscript/smartscript_web/py_checker/datacnt.txt",'r') as f:
        s = f.readline()
        return JsonResponse({'issues': s})

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def saveClient(request, name):
    client_ip = get_client_ip(request)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    with open('/home/smartscript/smartscript_web/py_checker/APICall/'+name+'.log', 'a') as f:
        f.write(str(cur_time)+' '+client_ip+'\n')
    

def upload(request):
    saveClient(request, "upload")
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return JsonResponse({'issues': "No file selected."})
        name = myFile.name
        destination = open(os.path.join("/home/smartscript/smartscript_web/static/py_checker/misc/upload","1.py"),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()
        return JsonResponse({'issues': "File uploaded."})# HttpResponse("upload over!")

def uploadType(request):
    saveClient(request, "uploadType")
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return JsonResponse({'issues': "No file selected."})
        name = myFile.name
        destination = open(os.path.join("/home/smartscript/smartscript_web/static/py_checker/misc/type","1.py"),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()
        return JsonResponse({'issues': "File uploaded."})# HttpResponse("upload over!")

def uploadKube(request):
    saveClient(request, "uploadKube")
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return JsonResponse({'issues': "No file selected."})
        name = myFile.name
        destination = open(os.path.join("/home/smartscript/smartscript_web/static/py_checker/misc/kube","1.yaml"),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()
        return JsonResponse({'issues': "File uploaded."})# HttpResponse("upload over!")

@csrf_exempt
def TypeModel(request):
    saveClient(request, "TypeModel")
    # try:
    time_start=time.time()
    # code = request.POST['code']  # this is for form encoded data
    code = json.loads(request.body)['code']
    model = int( str(json.loads(request.body)['model']))
    issues = None
    inconsistencies = ""
    if model == 0:
        issues = TypeModelServer.getResult(code)
    elif model==1:
        issues = newTypeModel.getResult(code)
    elif model==2:
        issues = PYIModel.getResult(code)
    elif model==3:
        issues,inconsistencies, marker = TypeContextDetector.Detector(code)
    elif model==4:
        issues,inconsistencies, marker = BaseContextDetector.Detector(code)
    else:
        return JsonResponse({'result': code,'issue':'No Model Selected.'})
    if isinstance(issues,str):
        return JsonResponse({'result': code,'issue':issues})
    codepiece = str(code).split("\n")
    codepiece2 = []
    cnt = 0
    for lineno,lcode in enumerate(codepiece): 
        lcode2 = lcode
        lineno = str(lineno+1)
        if lineno in issues:
            lcode2 = lcode2+" # "
            for varName,typeinfo in issues[lineno]:
                lcode2 = lcode2+" "+varName+":"+typeinfo+" |"
                cnt = cnt+1
        if len(lcode2)>0 and lcode2[-1]=='|':
            lcode2 = lcode2[:-1]
        codepiece2.append(lcode2)
    codepiece2 = "\n".join(codepiece2)
    # logger.debug(issues)
    time_end=time.time()
    if cnt==0:
        return JsonResponse({'result': codepiece2,'issue': 'No annotation found.'})
    if model==3 or model == 4:
        return JsonResponse({'result': codepiece2,'issue': 'Total time: '+str(cnt)+" variables analyzed in "+ str(round(time_end-time_start,3))+" s, "+"one variable at "+str(round((time_end-time_start)*1000/cnt,3))+" ms"+'\n'+inconsistencies,'marker':marker})    
    return JsonResponse({'result': codepiece2,'issue': 'Total time: '+str(cnt)+" variables analyzed in "+ str(round(time_end-time_start,3))+" s, "+"one variable at "+str(round((time_end-time_start)*1000/cnt,3))+" ms"+'\n'+inconsistencies})
    # except KeyError:
        # return JsonResponse({'result': code,'issue':"Key Error.\n"+str(KeyError)})





@csrf_exempt
def kubeModel(request):
    saveClient(request, "kubeModel")
    try:
        # code = request.POST['code']  # this is for form encoded data
        code = json.loads(request.body)['code']
        issues = kubeModelServer.main(code)
        logger.debug(issues)
        return JsonResponse({'issues': issues})
    except KeyError:
        return JsonResponse('')
@csrf_exempt
def smartModel(request):
    saveClient(request, "smartModel")
    try:
        # code = request.POST['code']  # this is for form encoded data
        code = json.loads(request.body)['code']
        issues = BPEModel.predict(code)
        logger.debug(issues)
        return JsonResponse({'issues': issues})
    except KeyError:
        return JsonResponse('')
def smart_script(request):
    saveClient(request, "smart_script")
    try:
        # code = request.POST['code']  # this is for form encoded data
        code = json.loads(request.body)['code']
        issues = checker.smart_script(code)
        logger.debug(issues)
        return JsonResponse({'issues': issues})
    except KeyError:
        return JsonResponse('')
def runCode(request):
    saveClient(request, "runCode")
    try:
        # code = request.POST['code']  # this is for form encoded data
        code = json.loads(request.body)['code']
        classtype = json.loads(request.body)['type']
        issues = checker.run_code(code,classtype)
        logger.debug(issues)
        return JsonResponse({'issues': issues})
    except KeyError:
        return JsonResponse('')
def check(request):
    saveClient(request, "check")
    try:
        # code = request.POST['code']  # this is for form encoded data
        code = json.loads(request.body)['code']
        issues = checker.check_code(code)
        logger.debug(issues)
        return JsonResponse({'issues': issues})
    except KeyError:
        return JsonResponse('')


def sample_code(request):
    saveClient(request, "sample_code")
    if request.method == 'GET':
        try:
            sample_id = int(request.GET.get('id'))
        except ValueError:
            return JsonResponse('')
        # if sample_id is None:
        #     return JsonResponse('')
        # if not isinstance(sample_id, int):
        #     return JsonResponse('')
        sample_folder = staticfiles_storage.path('py_checker/misc/{}'.format(sample_id))
        if not os.path.isdir(sample_folder):
            return JsonResponse('')
        code_files = os.listdir(sample_folder)
        if len(code_files) != 2:
            return JsonResponse('')
        code_ext = os.path.splitext(code_files[0])[-1]
        try:
            response = JsonResponse({
                'language': code_ext,
                "buggy": open(os.path.join(sample_folder, 'buggy' + code_ext)).read(),
                'fix': open(os.path.join(sample_folder, 'fix' + code_ext)).read()
            })
        except (IOError, FileNotFoundError):
            response = JsonResponse('')
        return response

def getUploadFile(request):
    if request.method == 'GET':
        sample_folder = staticfiles_storage.path('/home/smartscript/smartscript_web/static/py_checker/misc/upload/')
        if not os.path.isdir(sample_folder):
            return JsonResponse('')
        code_files = os.listdir(sample_folder)
        try:
            response = JsonResponse({
                "buggy": open(os.path.join(sample_folder, code_files[0])).read()
            })
        except (IOError, FileNotFoundError):
            response = JsonResponse('')
        return response

def getUploadTypeFile(request):
    if request.method == 'GET':
        sample_folder = staticfiles_storage.path('/home/smartscript/smartscript_web/static/py_checker/misc/type/')
        if not os.path.isdir(sample_folder):
            return JsonResponse('')
        code_files = os.listdir(sample_folder)
        try:
            response = JsonResponse({
                "buggy": open(os.path.join(sample_folder, code_files[0])).read()
            })
        except (IOError, FileNotFoundError):
            response = JsonResponse('')
        return response

def getUploadKubeFile(request):
    if request.method == 'GET':
        sample_folder = staticfiles_storage.path('/home/smartscript/smartscript_web/static/py_checker/misc/kube/')
        if not os.path.isdir(sample_folder):
            return JsonResponse('')
        code_files = os.listdir(sample_folder)
        try:
            response = JsonResponse({
                "buggy": open(os.path.join(sample_folder, code_files[0])).read()
            })
        except (IOError, FileNotFoundError):
            response = JsonResponse('')
        return response

