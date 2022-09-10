
# Individual Project 2
# Model-free Algorithms (Monte-Carlo and Temporal Difference)

#### Starting Date
* Week 5 Tuesday September 27, 2022 (23:59)

#### Due Date
* Week 7 Tuesday Oct 11, 2022 (23:59)

#### Total Points
* 100 (One Hundred)

## Goals
In this project, you will be asked to implement two model-free algorithms. The first one is Monte-Carlo(MC), including  the first visit of on-policy MC prediction and on-policy MC control for [blackjack](https://www.gymlibrary.dev/environments/toy_text/blackjack/). The second one is Temporal-Difference(TD), including Sarsa(on-policy) and Q-Learning(off-policy) for [cliffwalking](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/).

<img src="/Project2/img/project2-1updated.png" width="80%" >
<img src="/Project2/img/project2-2updated.png" width="80%" >

## Guidelines
* Implement functions in mc.py and td.py
* Evaluate functions by typing "nosetests -v mc_test.py" and "nosetests -v td_test.py" respectively, or try "python3 -m nose -v mc_test.py" and "python3 -m nose -v td_test.py". (put your function file and test file into the same folder).
* Note: If your code is correct, it will pass all the tests in most cases, but it may fail sometimes (because epsilon-greedy policy will produce randomness), so once you codes fail, please run the test file three times to see if the failure is caused by policy randomness or by your wrong code. (The correct code usually passes the all the tests within 3 times runing).
* <span style="color:blue">**TA will run your code three times. You will get full credits if one of the tests passes.**</span><br/>

## Deliverables

Please compress your mc.py and td.py files into a zip file (firstName_lastName_hw2.zip) and submit it to Canvas.

## Grading

* Initial policy (2 points)
* Epsilon greedy policy (8 points)
* MC prediction (first visit, on-policy) (20 points)
* MC control (first visit, on-policy) (20 points)
* Sarsa (25 points)
* Q-Learning (25 points)<br/>


## Hints
* On-policy first visit Monte-Carlo prediction
<img src="/Project2/img/mc_predict.png" width="80%" >

* On-policy first visit Monte-Carlo control
<img src="/Project2/img/mc.png" width="80%" >

* Sarsa (on-policy TD control)
<img src="/Project2/img/sarsa.png" width="80%" >

* Q-learing (off-policy TD control)
<img src="/Project2/img/q-learning.png" width="80%" >




