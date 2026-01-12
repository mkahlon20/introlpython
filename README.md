# AD450
This is the repo for AD450 Data Science Development.

To use this repo follow the directions in `Python, VSCode and Github Setup` on Canvas.

Then follow the directions below to set up this repo.


## Anaconda Virtual Environment
Create an Anaconda Environment using command line. You can also create one using Anaconda Navigator but the yml file is still required so that when I grade it and create an Anaconda environment, it will automatically install all libraries I need.

```
conda env create -f requirements.yml
conda activate my_env
```

Confirm all dependencies:
```
flake8 --version
black --version
nbqa --version
```

Note: nbqa is a library that lets us use linters on ipynb files.

At this point, you will be within your Anaconda virtual environment within Terminal. To use it as a Python interpreter in VScode, you will want to access the command bar with Ctrl+Shift+P and then Select Python Interpreter. You should see your Anaconda virtual environmant name listed.s

How to use Linters
Flake8 will tell us what the problems are. Black will automatically fix our issues.
Flake8:
Lets say we have this Python code:
```
x = [1, 2, 3]
for i in x: print(i, end=", ")
```

If we run: nbqa flake8 test.ipynb

We will get:
```
(my_env) [Eric-Lloyd] > nbqa flake8 test.ipynb
test.ipynb:cell_1:2:11: E701 multiple statements on one line (colon)
```

This tells us that we should not have multiple statements on one line. 

Black: 
We can fix this manually or we can use Black to autofix it for us.
```
(my_env) [Eric-Lloyd] > nbqa black test.ipynb 
reformatted test.ipynb
 
All done! ✨ 🍰 ✨
1 file reformatted.
```

Now our code should look like:
```
x = [1, 2, 3]
for i in x:
    print(i, end=", ")
```

IMPORTANT: When using VScode, you must save regularly in order to see changes! It does not auto-save.
There is a way to set up VSCode to autoformat using Black when you save. You are free to explore that if you wish. For now, the given instructions are around using the command line manually. 
I strongly suggest (because I cannot monitor this reasonably), that you use Flake8 to see what needs to be changed and understand what the rules are for standardized good Python coding and then use Black to autoformat manually. The most important part is using Flake8 to understand so that eventually, you are coding up to standard.

This is a secret text added for git reasons.