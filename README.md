# TBPS-Team6
This is the repository for Year 3 TBPS 2022 team 6

### Suggestion for file path
To avoid changing the path for data files each time we sync the python code, we can place them in the same folder so that the following works
```
import os

file_name = 'signal.pkl'
content = pd.read_pickle(os.getcwd() + f'\year3-problem-solving' + f'\{file_name}')
content.head()
```
where `year3-problem-solving` is the file from Mitesh, downloaded from [here](https://imperialcollegelondon.app.box.com/s/mwdgg4uz7hdz56bx6w4loc04qvzb7tmy).

By the same folder, I mean this:

![image](https://user-images.githubusercontent.com/97897047/150685271-84552dd0-0f77-43a6-9484-0c57967a8028.png)

**Note**: you'll need to change your working directory to the folder containing your Python/datasets. Otherwise, your "run" will work but the console might not. I used Pycharm for IDE.
