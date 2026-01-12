# Making PRs

This is instructions on how to make PRs (Pull Requests) for submissions of the weekly problems in AD450. 

1. Follow the directions in for [Configure Github with VSCode](https://canvas.seattlecolleges.edu/courses/30109/pages/python-vscode-and-github-setup) to set up this repo. Be sure to follow the directions for copying this repo using the template button. 

2. First make a small edit to a file and save it. Then you will check the status of your github repo by running `git status`. You will run this a lot in this tutorial. This is what I got. Notice I am on the main branch, I made modifications to `week_2_intro_ipython.ipynb` and I have added the file `week_4_panadas.ipynb`. We are going to be working with just the week 2 file in this example.

<img width="555" alt="Screenshot 2025-01-08 at 10 32 02 AM" src="https://github.com/user-attachments/assets/bbdf3c21-12da-42dc-9eb6-788a4392a026" />

3. Then create a branch by running the command below with BRANCH_NAME replaced with a reasonable file name for the branch. In the screenshot below I used `week_2_ipython_branch`. I also checked the status of my branch after. You see I am no longer on main but now on `week_2_ipython_branch`.
```
git checkout -b BRANCH_NAME
```

<img width="581" alt="Screenshot 2025-01-08 at 10 33 04 AM" src="https://github.com/user-attachments/assets/abdbc01b-c2a4-44ae-9816-6029aff43967" />

4. Now you will stage the files on your branch with the add command below. Checking the status you can see that not the week 2 file is green. This means it is staged on the branch. The week 4 file is still red because we did not add it to the branch.

```
git add week_2_intro_ipython.ipynb
```

<img width="559" alt="Screenshot 2025-01-08 at 10 33 25 AM" src="https://github.com/user-attachments/assets/f47aacab-cfb9-4c0b-b22b-156ebdc2db60" />


5. Next we will commit the changes with a message. The message is added with the flag `-m` and the message will be whatever follows in quotes.
```
git commit -m "Added first attempt of week 2 problems"
```

<img width="670" alt="Screenshot 2025-01-08 at 10 33 55 AM" src="https://github.com/user-attachments/assets/4786acdf-4729-427a-b8ba-164d9a04180d" />


6. We will now push our branch to github. I first run `git push`. This will not work the first time to push a new branch but it will give you the command you need. You can see in the output below git gave me the command I needed to push my branch. Yours might look different if you named your branch something else. Run the command git give you and you should get a link. This will be the PR that you open, click the link.

<img width="639" alt="Screenshot 2025-01-08 at 10 34 28 AM" src="https://github.com/user-attachments/assets/ba8d985d-45b7-47b1-a6b9-80e98ea4201f" />


7. This link will open a PR on your branch in your repo. It should be automatically filled out with the PR template in this repo. You will fill out this template before opening your PR.

<img width="1437" alt="Screenshot 2025-01-08 at 11 30 27 AM" src="https://github.com/user-attachments/assets/939751af-c67b-455b-bddc-9b3363bc1156" />


8. Now let's suppose you made some changes to your code that you want to appear in your PR. Once you have saved your changes run `git status` to see your unstages changes. You will follow steps 4, 5, and 6 again. In step 5 you will use a different commit message to describe your new changes. In step 6 you will just need to run `git push` since you've already created the branch on GitHub.

<img width="597" alt="Screenshot 2025-01-08 at 10 35 26 AM" src="https://github.com/user-attachments/assets/8831e54a-1cbf-4154-90ae-c4365f1c29bd" />

<img width="541" alt="Screenshot 2025-01-08 at 10 35 46 AM" src="https://github.com/user-attachments/assets/ee894ba2-06d6-4947-b1e1-54bcd8456894" />

<img width="664" alt="Screenshot 2025-01-08 at 10 36 24 AM" src="https://github.com/user-attachments/assets/f7ea84fc-1a83-4762-8f4c-22e2de077ed9" />

<img width="600" alt="Screenshot 2025-01-08 at 10 36 42 AM" src="https://github.com/user-attachments/assets/de682a2a-dea5-4fad-9a92-c77a9d66b28a" />


9. Once you have commited all the code to your branch and pushed it to GitHub you are ready to open your PR. You'll fill out the PR tempated and then click the `Create pull request` button. Once you have done that you will add your instructor and TA as reviewers and submit the link to PR to the assignment in Canvas. If you don't know the GitHub username of the instructor and TA just submit the link to canvas and we will add ourselves as reviewers.

**Congrats on making your first PR!**
