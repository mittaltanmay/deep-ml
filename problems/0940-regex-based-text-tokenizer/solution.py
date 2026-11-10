import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    ans=[]
    temp=""
    s=text
    i=0
    while i<len(text):
        if(text[i]==' '):
            if temp!="":
                ans.append(temp)
            temp=""
            i=i+1
            continue
        elif s[i]=='.' or s[i]==',' or s[i]==':' or s[i]==';' or s[i]=='!' or s[i]=='"' or s[i]=='(' or s[i]==')' or s[i]=="'" or s[i]=='?' or s[i]=="_":
            if temp!="":
                ans.append(temp)
            ans.append(s[i])
            temp=""
            i=i+1
            continue
        elif i+1<len(text) and s[i]=='-' and s[i+1]=='-':
            if temp!="":
                ans.append(temp)
            ans.append("--")
            temp=""
            i=i+2
            continue
        elif s[i] in ['\t','\n'] :
            if(temp!=""):
                ans.append(temp)
            temp=""
            i=i+1
            continue 
        temp=temp+s[i]
        i=i+1
    if temp!="":
        ans.append(temp)
    return ans
        
            
