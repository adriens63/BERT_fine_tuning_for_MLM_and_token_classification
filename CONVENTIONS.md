# Alignment conventions  




## Lign breaks :
A single line between each party within the same function, 
three lines between functions within the same class, 
four lines between classes or independent functions. 




## Capitals :  
Capital-letter constants,  
function and variables in snake_case,  
class and exceptions in CamelCase.  



## Spaces :  
A space between operations: 3 + 2,  
and after the commas.  



## Presentation of functions:  
Type-hint added this way,  
``` 
def __init__(self, k:int) -> None:
        pass
``` 
description of the function just after the def in double quotes, not with # in the middle in "clean" versions,  
single quotes instead of double quotes for strings.  



## Execution of scripts:  
Execute from analysis_of_textual_job_descriptions_with_Pole_Emploi directory with  
``` 
python -m src.format
```
That way all the relative paths are from this directory.  