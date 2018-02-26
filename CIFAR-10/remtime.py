
def printTime(remtime):
	hrs = int(remtime)/3600
	mins = int((remtime/60-hrs*60))
	secs = int(remtime-mins*60-hrs*3600)
	print("########  "+str(hrs)+"Hrs "+str(mins)+"Mins "+str(secs)+"Secs remaining  ########")