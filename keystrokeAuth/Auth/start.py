# -*- coding: utf-8 -*- 
import os,time
import datadeal
from threading import Thread

currdir = ""
dirpath = "./data/usr_txt/"
resl = None
datalog=[]

def datatest():
	global resl,datalog
	retn = datadeal.datadeal(datalog)
	if resl == None:
		resl = retn
	elif resl != retn:
		resl = retn
		datalog = []
		return
	if resl==True:
		print("PASS!")
	else:
		print("DENY!")


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def call_hook():
	hook_exe = "hook_Service.exe"
	os.system(hook_exe)
	return 0
	
def find_curr_dir():
	global currdir,dirpath
	while(True):
		plist = os.listdir(dirpath)
		plist.sort(reverse=True)
		if len(plist)>0 and currdir != plist[0]:
			currdir = plist[0]
		

del_file(dirpath)
thook = Thread(target=call_hook)
thook.start()
thdir = Thread(target=find_curr_dir)
thdir.setDaemon(True) 
thdir.start()
time.sleep(1)
print("认证开始...")

while(currdir==""):
	pass
keysdir = currdir
f = open(dirpath+keysdir,'r')

while(thook.isAlive()):
	if keysdir != currdir:		
		f.close()
		Thread(target=datatest).start()
		datalog = []
		keysdir = currdir
		f = open(dirpath+keysdir,'r')
	line = f.readline()
	if(line!=''):
		keys = line.replace('\n','').split('\t')
		if(len(keys)==3):
			datalog.append([int(keys[0]),int(keys[1]),int(keys[2])])
			if(len(datalog)==31):
				Thread(target=datatest).start()
				datalog = datalog[20:]
if datalog!=[]:
	Thread(target=datatest).start()
	
		