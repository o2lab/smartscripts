from django.urls import path

from . import views
 
urlpatterns = [
    path('', views.index, name='index'),
    path('pychecker', views.pychecker, name='pychecker'),
    path('Report', views.Report, name='Report'),
    path('echarts', views.echarts, name='echarts'),
    path('linecharts', views.linecharts, name='linecharts'), 
    path('check', views.check, name='check'),
    path('runCode', views.runCode, name='runCode'),
    path('sampleCode', views.sample_code, name='sample_code'),
    path('smartScript', views.smart_script, name='smart_script'),
    path('smartModel', views.smartModel, name='smartModel'),
    path('model', views.model, name='model'),
    path('upload', views.upload, name='upload'),
    path('API', views.api, name='api'),
    path('bug', views.bug, name='bug'),
    path('getUploadFile', views.getUploadFile, name='getUploadFile'),
    path('clcheck', views.clcheck, name='clcheck'),
    path('datacnt', views.datacnt, name='datacnt'),
    path('githook', views.githook, name='githook'),
    path('reload', views.reload, name='reload'),

    path('typeChecker', views.typeChecker, name='typeChecker'),
    path('typeGenerator', views.typeGenerator, name='typeGenerator'),
    path('smartkube', views.smartkube, name='smartkube'),
    path('uploadType', views.uploadType, name='uploadType'),
    path('uploadKube', views.uploadKube, name='uploadKube'),
    path('TypeModel', views.TypeModel, name='TypeModel'),
    path('kubeModel', views.kubeModel, name='kubeModel'),
    path('getUploadTypeFile', views.getUploadTypeFile, name='getUploadTypeFile'),
    path('getUploadKubeFile', views.getUploadKubeFile, name='getUploadKubeFile'),
]
