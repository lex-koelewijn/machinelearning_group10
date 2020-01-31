# Machine Learning Project by Group 10
Finishing Bach's unfinished fugue
### Group members
* Lex Koelewijn
* Dirk Jelle Schaap
* Boris Luttikhuizen
* Joppe Boekestijn


## Run Instructions

Please create a virtual environment in which Python version 3.6.7 is used and activate it.

Clone this repository and navigate to its local folder.
```
git clone https://github.com/JoppeBoekestijn/machinelearning_group10.git
cd machinelearning_group10/
```
Install all required packages using pip.
```
pip install -r requirements.txt
```
If you want to be able to run the conversion to a ```.wav``` file, install FluidSynth too. You must be using Linux in that case.
```
sudo apt install fluidsynth
```
If you would like to run the code we have used to learn and predict, stay in ```machinelearning_group10/```.

If you would like to convert the output of the prediction into a ```.wav``` file, navigate into the subfolder ```midi/```.

In either case, run the script called ```jupyter_script.sh```.
```
bash jupyter_script.sh
```
Select the ```code.py``` file in the interface by typing its number and pressing enter.

Follow the URL that is given in the console. You should now see the notebook file opened, otherwise open it manually in the file explorer on the left side of the screen. Run a cell of code using ```CTRL + ENTER```.
