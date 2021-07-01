#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
def updateWeb(): 
    DIR = "./coll_stat/"
    print("Updating")
    # def compare(x, y):
    #     stat_x = os.stat(DIR + "/" + x)
    #     stat_y = os.stat(DIR + "/" + y)
    #     if stat_x.st_ctime < stat_y.st_ctime:
    #         return -1
    #     elif stat_x.st_ctime > stat_y.st_ctime:
    #         return 1
    #     else:
    #         return 0


    # In[2]:


    iterms = os.listdir(DIR)


    # In[3]:


    iterms.sort()


    # In[4]:


    # iterms[1]


    # In[5]:


    MaxDB = 100
    MaxCol = 100000


    # In[6]:


    dbName = []
    CollectData = []
    data = [[0 for i in range(MaxCol)] for j in range(MaxDB)]
    cur = 0
    banList = ['pkg_dependents','test','read_bug']

    # In[7]:


    # str(dbName)


    # In[8]:


    # data[10][1]


    # In[9]:


    for item in iterms:
        if item[-1]=='v':# is a csv file
            fdir = DIR+item
            with open(fdir,"r") as f:
                CollectData.append(item[10:24])
                for line in f:
                    if line[0]=='b':# bug_db
                        dat = line.split(",")
                        name = dat[0].split(".")[-1]
                        count = dat[3]
            #             print(name,count)
                        if name in banList:
                            continue
                        if name not in dbName:
                            dbName.append(name)
                        data[dbName.index(name)][cur] = int(count)
            cur += 1


    # In[10]:


    # item[10:24]
    # https://echarts.apache.org/examples/en/editor.html?c=line-stack&theme=dark
    # https://echarts.apache.org/examples/en/editor.html?c=area-stack&theme=dark


    # In[11]:


    # fdir = DIR+item
    # with open(fdir,"r") as f:
    #     CollectData.append(item[10:24])
    #     for line in f:
    #         if line[0]=='b':# bug_db
    #             dat = line.split(",")
    #             name = dat[0].split(".")[-1]
    #             count = dat[3]
    # #             print(name,count)
    #             if name not in dbName:
    #                 dbName.append(name)
    #             data[dbName.index(name)][cur] = int(count)
    # cur += 1


    # In[12]:


    headstr= """
    {% load static %}
    {% load bootstrap4 %}
    {% bootstrap_css %}
    {% bootstrap_javascript jquery='full' %}

    <!DOCTYPE html>
    <html style="height: 100%">
    <head>
        <meta charset="utf-8">
    </head>
    <body style="height: 100%; margin: 0">
    <input id="selectall" type="button" class="btn btn-primary" value="Select None" />
        <div id="container" style="height: 100%"></div>
        <script src="{% static 'py_checker/js/echarts.min.js' %}"></script>
        <script src="{% static 'py_checker/js/dark.js' %}"></script>

        <!-- <script src="{% static 'py_checker/js/shine.js' %}"></script>
        <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/echarts.min.js"></script>
        <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts-gl/echarts-gl.min.js"></script>
        <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts-stat/ecStat.min.js"></script>
        <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/extension/dataTool.min.js"></script>
        <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/map/js/china.js"></script>
        <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/map/js/world.js"></script>
        <script type="text/javascript" src="https://api.map.baidu.com/api?v=2.0&ak=xfhhaTThl11qYVrqLZii6w8qE5ggnhrY&__ec_v__=20190126"></script>
        <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/extension/bmap.min.js"></script>
        <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/simplex.js"></script> -->
        <script type="text/javascript">
    var dom = document.getElementById("container");
    var myChart = echarts.init(dom,'dark');
    var app = {};
    option = null;
    """

    tailstr = """
    ]
    };
    ;
    if (option && typeof option === "object") {
        myChart.setOption(option, true);
    }
    
    
    var selectArr = myChart.getOption().legend[0].data;
        
        $('#selectall').click(function(){
            var flag = $(this).attr('flag');
            if(flag == 1){
                var val = true;
                $(this).attr('flag',0);
                $(this).val('Select None');
            }else{
                var val = false;
                $(this).attr('flag',1);
                $(this).val('Select All');
            }
            var obj = {};
            for(var key in selectArr){
                obj[selectArr[key]] = val;
            }    
            option.legend.selected = obj;
            myChart.setOption(option);
        });
        </script>
    </body>
    </html>

    """


    # In[13]:


    optionstr = """
    option = {
        title: {
            text: 'Line-stack Graph'
        },
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            type: 'scroll',
            // orient: 'vertical',
            left: 80,
            right: 80,
            top: 20,
            bottom: 20,
            data:[
    """


    # In[14]:


    # str(dbName)[1:-1]


    # In[15]:


    optionstr+=str(dbName)[1:-1]+"""
    ]
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        toolbox: {
            feature: {
                saveAsImage: {}
            }
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: ["""


    # In[16]:


    optionstr+=str(CollectData)[1:-1]+"""

    ]
        },
        yAxis: {
            type: 'value'
        },
        series: [
    """


    # In[17]:


    allstrdb = ""
    for db in dbName:
    # db = dbName[0]
        dbdata=str(data[dbName.index(db)][:cur])[1:-1]
        strdb ="""
        {
                    name:'%(name)s',
                    type:'line',
                    stack: 'total',
                    data:[%(data)s]
                },
        """ % {'name':db,'data':dbdata}
        allstrdb += strdb

    # allstrdb+=strdb.format(name=db,data=dbdata)


    # In[18]:


    # str(dbdata)[1:-1]


    # In[19]:


    finalstr = headstr + optionstr + allstrdb + tailstr


    # In[20]:


    # strdb


    # In[21]:


    with open("/home/smartscript/smartscript_web/py_checker/templates/py_checker/line-stack.html","w") as f:
        f.write(finalstr)
        f.flush()

    # print(finalstr)
    totalcnt = 0
    for i in range(len(dbName)):
        totalcnt+=data[i][cur-1]
    with open("/home/smartscript/smartscript_web/py_checker/datacnt.txt",'w') as f:
        f.write(str(totalcnt))
        f.flush()
    
    # In[ ]:



    ########### Echarts ###############

    echartsstr = """

    {% load static %}
{% load bootstrap4 %}
{% bootstrap_css %}
{% bootstrap_javascript jquery='full' %}

<!DOCTYPE html>
<html style="height: 100%">
   <head>
       <meta charset="utf-8">
   </head>
   <body style="height: 100%; margin: 0">
   <input id="selectall" type="button" class="btn btn-primary" value="Select None" />
       <div align="center" id="container" style="height: 100%"></div>
       <script src="{% static 'py_checker/js/echarts.min.js' %}"></script>
       <script src="{% static 'py_checker/js/dark.js' %}"></script>
       <script src="{% static 'py_checker/js/shine.js' %}"></script>
       <!-- <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/echarts.min.js"></script> -->
       <!-- <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts-gl/echarts-gl.min.js"></script>
       <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts-stat/ecStat.min.js"></script>
       <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/extension/dataTool.min.js"></script>
       <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/map/js/china.js"></script>
       <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/map/js/world.js"></script>
       <script type="text/javascript" src="https://api.map.baidu.com/api?v=2.0&ak=xfhhaTThl11qYVrqLZii6w8qE5ggnhrY&__ec_v__=20190126"></script>
       <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/echarts/extension/bmap.min.js"></script>
       <script type="text/javascript" src="http://echarts.baidu.com/gallery/vendors/simplex.js"></script> -->
       <script type="text/javascript">
var dom = document.getElementById("container");
var myChart = echarts.init(dom,'dark');
var app = {};
option = null;
option = {
    title : {
        text: 'Database Information',
        subtext: 'SmartScripts.org',
        x:'center'
    },
    tooltip : {
        trigger: 'item',
        formatter: "{a} <br/>{b} : {c} ({d}%)"
    },
    legend: {
        type: 'scroll',
        orient: 'vertical',
        // left: 'right',
        right: 30,
        top: 50,
        bottom: 50,
        data: ["""
    
    echartsstr += str(dbName)[1:-1]+"""
    ]
    },
    series : [
        {
            name: 'Dataset Information',
            type: 'pie',
            radius : '55%',
            center: ['50%', '60%'],
            data:["""
    for name in dbName:
        num = data[dbName.index(name)][cur-1]
        fstr = "{value: %(num)s , name:'%(name)s'},\n" % {'num':str(num),'name':name}
        echartsstr+=fstr
    echartsstr +="""
    ],
            itemStyle: {
                emphasis: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
            }
        }
    ]
};
;
if (option && typeof option === "object") {
    myChart.setOption(option, true);
}

    var selectArr = myChart.getOption().legend[0].data;
        
        $('#selectall').click(function(){
            var flag = $(this).attr('flag');
            if(flag == 1){
                var val = true;
                $(this).attr('flag',0);
                $(this).val('Select None');
            }else{
                var val = false;
                $(this).attr('flag',1);
                $(this).val('Select All');
            }
            var obj = {};
            for(var key in selectArr){
                obj[selectArr[key]] = val;
            }    
            option.legend.selected = obj;
            myChart.setOption(option);
        });

       </script>
   </body>
</html>
"""
    with open("/home/smartscript/smartscript_web/py_checker/templates/py_checker/echarts.html","w") as f:
        f.write(echartsstr)
        f.flush()

if __name__=="__main__":
    updateWeb()

